import os
import pandas as pd
import requests
import shutil
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Dati caricati dai file CSV
images_obs_df = pd.read_csv("images_observations.csv", delimiter='\t')
observations_df = pd.read_csv("observations.csv", delimiter='\t')
names_df = pd.read_csv("names.csv", delimiter='\t')

# Correzioni manuali per alcuni synonym_id
manual_corrections = {
    '10207': "Panaeolus subbalteatus", '609': "Conocybe rugosa", '696': "Biatora pallens",
    '9564': "Hemimycena albicolor", '9192': "Fulgidea sierrae", '7805': "Limacella guttata",
    '7668': "Lactarius pterosporus", '7803': "Athelia salicum", '8840': "Gliophorus viscaurantius",
    '7678': "Lactarius fennoscandicus", '5062': "Morchella conica", '7506': "Orbilia delicatula",
    '7932': "Hohenbuehelia podocarpinea", '9990': "Morchella costata", '5588': "Psathyrellaceae",
    '7923': "Hourangia pumila", '5101': "Piptoporus australiensis", '4851': "Isaria surinamensis",
    '8841': "Gliophorus sulfureus", '3378': "Russula emetica group", '7829': "Hypochnicium sphaerosporum",
    '9576': "Inocybe flavella", '7369': "Lacrymaria pyrotricha", '9696': "Styrofomes riparius",
    '5710': "Mycolindtneria trachyspora", '7493': "Pholiota jahnii", '8549': "Arthopyrenia epidermidis",
    '8808': "Nectria fuckeliana", '2771': "Geastrum arenarium", '7761': "Kretzschmariella culmorum",
    '5370': "Suillus", '9577': "Mallocybe heimii", '7405': "Typhula uncialis", '705': "Xanthoconium affine"
}

extensions = ['.jpg', '.jpeg', '.png', 'raw', 'tiff', 'heif', 'dng', 'bmp', 'gif', 'psd']
log_errors = []

def sanitize_folder_name(text):
    return text.translate(str.maketrans({"<": "", ">": "", ":": "", "\"": "", "\\": "", "|": "", "?": "", "*": ""}))

def determine_folder(text_name):
    if (text_name.lower().endswith('series') or text_name.lower().endswith('group') or 
        'sect.' in text_name or len(text_name.split()) == 1):
        folder_name = os.path.join('Unknown species', text_name)
    else:
        parts = text_name.split()
        first_two = ' '.join(parts[:2]) if len(parts) >= 2 else text_name
        folder_full_name = f"{text_name} species" if not any(x in text_name for x in ['var.', 'f.', 'subsp.']) else text_name
        folder_name = os.path.join('Known species', sanitize_folder_name(first_two), sanitize_folder_name(folder_full_name))
    return folder_name

def find_image_locally(image_id):
    for root, _, files in os.walk("MO/"):
        for file in files:
            if any(file == f"{image_id}{ext}" for ext in extensions):
                current_path = os.path.join(root, file)
                return current_path
    return None

def check_image_integrity(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifica integrità dell'immagine
        return True
    except (UnidentifiedImageError, IOError):
        os.remove(image_path)  # Elimina immagine corrotta
        print(f"Corrupted image {image_path} removed.")
        return False

def resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Convertiamo in RGB per evitare formati incompatibili
            width, height = img.size
            min_side = min(width, height)
            if min_side > 720:
                scaling_factor = 720 / min_side
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(image_path)
    except Exception as e:
        log_errors.append(f"Error resizing {image_path}: {e}")

def check_and_download_image(image_id, target_path):
    for ext in extensions:
        url = f"https://storage.googleapis.com/mo-image-archive-bucket/orig/{image_id}{ext}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            print(f"Error downloading {image_id}{ext}: {e}")
    return False

def process_images():
    for idx, row in tqdm(images_obs_df.iterrows(), total=images_obs_df.shape[0], desc="Processing images"):
        image_id = row['image_id']
        observation_id = row['observation_id']
        observation_row = observations_df[observations_df['id'] == observation_id]
        
        if observation_row.empty:
            print(f"Image ID {image_id} has no corresponding observation. Deleting...")
            os.remove(f"MO/{image_id}")
            continue

        name_id = observation_row.iloc[0]['name_id']
        name_row = names_df[names_df['id'] == name_id]
        
        if name_row.empty:
            continue
        
        synonym_id = name_row.iloc[0]['synonym_id']
        correct_spelling_id = name_row.iloc[0]['correct_spelling_id']
        
        text_name = None
        if pd.notna(synonym_id) and synonym_id in manual_corrections:
            text_name = manual_corrections[synonym_id]
        elif pd.notna(synonym_id) and synonym_id.isnumeric():
            candidates = names_df[(names_df['synonym_id'] == int(synonym_id)) & (names_df['deprecated'] == 0)]
            if not candidates.empty:
                text_name = min(candidates['text_name'], key=len)
            else:
                log_errors.append(f"All names with synonym_id {synonym_id} are deprecated for image ID {image_id}")
        elif pd.notna(correct_spelling_id) and correct_spelling_id.isnumeric():
            synonym_candidates = names_df[(names_df['id'] == int(correct_spelling_id))]
            synonym_id = synonym_candidates.iloc[0]['synonym_id'] if not synonym_candidates.empty else None
            candidates = names_df[(names_df['synonym_id'] == synonym_id) & (names_df['deprecated'] == 0)]
            text_name = min(candidates['text_name'], key=len) if not candidates.empty else name_row.iloc[0]['text_name']
        else:
            text_name = name_row.iloc[0]['text_name']
        
        folder_name = determine_folder(sanitize_folder_name(text_name))
        target_path = os.path.join("MO", folder_name, f"{image_id}.jpg")

        local_path = find_image_locally(image_id)
        if local_path:
            print(f"Image {image_id} found at {local_path}. Moving to {target_path}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if check_image_integrity(local_path):
                shutil.move(local_path, target_path)
                resize_image(target_path)
            else:
                print(f"Image {local_path} is corrupted. Skipping...")
        else:
            print(f"Image {image_id} not found locally. Attempting download...")
            if check_and_download_image(image_id, target_path):
                resize_image(target_path)
            else:
                print(f"Failed to download image {image_id}")

# Esegui il processo
process_images()

# Log degli errori
with open("error_log.txt", "w") as log_file:
    for error in log_errors:
        log_file.write(error + "\n")

print("Processing complete. Check error_log.txt for issues.")
