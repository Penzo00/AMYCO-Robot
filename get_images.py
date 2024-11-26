#%%
import pandas as pd
import os
import requests
from tqdm import tqdm
import re

# Load CSV files
print("Loading CSV files...")
names_df = pd.read_csv("CSV/names.csv", sep="\t").drop(['author', 'rank'], axis=1).fillna(-1)
observations_df = pd.read_csv("CSV/observations.csv", sep="\t").drop(['when', 'location_id', 'lat', 'lng', 'vote_cache', 'is_collection_location', 'thumb_image_id'], axis=1).fillna(-1)
images_observations_df = pd.read_csv("CSV/images_observations.csv", sep="\t").fillna(-1)
images_observations_df[['image_id', 'observation_id']] = images_observations_df[['image_id', 'observation_id']].astype('int32')
observations_df[['id', 'name_id']] = observations_df[['id', 'name_id']].astype('int32')
names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']] = names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']].astype('int32')

# Define folders and logging paths
download_log_path = "download_errors.log"
recovery_file_path = "download_recovery.txt"
MO_folder = "MO/"
unknown_folder = os.path.join(MO_folder, "Unknown species/")
known_folder = os.path.join(MO_folder, "Known species/")

# Create base folders
os.makedirs(unknown_folder, exist_ok=True)
os.makedirs(known_folder, exist_ok=True)

# Headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

synonyms_ids = names_df['synonym_id'].unique()

# Manual dictionary for NULL synonym_id cases
manual_synonyms = {
    '10207': "Panaeolus subbalteatus",
    '5512': "Phaeotremella foliacea",
    '609': "Conocybe rugosa",
    '696': "Biatora pallens",
    '9564': "Hemimycena albicolor",
    '9192': "Fulgidea sierrae",
    '7805': "Limacella guttata",
    '7668': "Lactarius pterosporus",
    '7803': "Athelia salicum",
    '8840': "Gliophorus viscaurantius",
    '7678': "Lactarius fennoscandicus",
    '5062': "Morchella conica",
    '7506': "Orbilia delicatula",
    '7932': "Hohenbuehelia podocarpinea",
    '9990': "Morchella costata",
    '5588': "Psathyrellaceae",
    '7923': "Hourangia pumila",
    '5101': "Piptoporus australiensis",
    '4851': "Isaria surinamensis",
    '8841': "Gliophorus sulfureus",
    '3378': "Russula emetica group",
    '7829': "Hypochnicium sphaerosporum",
    '9576': "Inocybe flavella",
    '7369': "Lacrymaria pyrotricha",
    '9696': "Styrofomes riparius",
    '5710': "Mycolindtneria trachyspora",
    '7493': "Pholiota jahnii",
    '8549': "Arthopyrenia epidermidis",
    '8808': "Nectria fuckeliana",
    '2771': "Geastrum arenarium",
    '7761': "Kretzschmariella culmorum",
    '5370': "Suillus",
    '9577': "Mallocybe heimii",
    '7405': "Typhula uncialis",
    '705': "Xanthoconium affine",
    '7664': "Lactarius zonarius",
    '4348': "Amanita subnudipes",
    '4414': "Calosphaeria cyclospora",
    '8796': "Amanita alliodora",
    '1053': "Catapyrenium compactum",
    '4720': "Xylaria nigripes",
    '622': "Agaricus litoralis",
    '7359': "Lentinus dicholamellatus",
    '8786': "Hohenbuehelia reniformis",
    '7029': 'Agaricaceae',
    '9303': "Amauroderma yunnanense",
    '7844': "Hygrocybe sect. Pseudofirmae",
    '7208': "Entoloma umbiliciforme",
    '7108' :"Fungi"
}

def log_error(message):
    """Log errors to a file."""
    with open(download_log_path, "a") as log_file:
        log_file.write(message + "\n")
        
def save_progress(image_id):
    with open(recovery_file_path, "w") as recovery_file:
        recovery_file.write(str(image_id))

def load_last_downloaded():
    if os.path.exists(recovery_file_path):
        with open(recovery_file_path, "r") as recovery_file:
            return int(recovery_file.read().strip())
    return None
        
def expand_dataframe(initial_df, names_df):
    prev_len = 0  # Length of the DataFrame in the previous iteration

    while len(initial_df) > prev_len:  # Continue until no new rows are added
        prev_len = len(initial_df)  # Update previous length
        
        # Find all unique synonym_id and correct_spelling_id in the current DataFrame
        synonym_ids = initial_df['synonym_id'].unique()
        correct_spelling_ids = initial_df['correct_spelling_id'].unique()
        
        # Find new rows with corresponding synonym_id or correct_spelling_id and remove -1 values
        new_synonyms = names_df[names_df['synonym_id'].isin(synonym_ids) & (names_df['synonym_id'] >= 0)]
        new_correct_spellings = names_df[names_df['id'].isin(correct_spelling_ids) & (names_df['id'] >= 0)]
        
        # Combine new rows with the initial DataFrame and remove duplicates
        initial_df = pd.concat([initial_df, new_synonyms, new_correct_spellings], axis=0).drop_duplicates().reset_index(drop=True)
    
    return initial_df

# Carica l'ultimo punto di download completato
last_downloaded = load_last_downloaded()
start_index = 0

# Cerca l'ultimo indice dal quale riprendere il download
if last_downloaded is not None:
    start_index = images_observations_df[images_observations_df['image_id'] == last_downloaded].index[0] + 1

print("Downloading images...")

for i in tqdm(range(start_index, len(images_observations_df)), desc="Downloading images"):
    image_id = images_observations_df['image_id'].iloc[i]
    observation_id = images_observations_df['observation_id'].iloc[i]
    name_id = observations_df[observations_df['id'] == observation_id]['name_id'].values[0]
    text_name_pd = names_df[names_df['id'] == name_id]
    synonym_id = text_name_pd['synonym_id'].values[0]
    correct_spelling_id = names_df[names_df['id'] == name_id]['correct_spelling_id'].values[0]
    if synonym_id >= 0:
        synonyms = names_df[names_df['synonym_id'] == synonym_id]
        text_name_pd = pd.concat([text_name_pd, synonyms], axis=0).drop_duplicates()
    if correct_spelling_id >= 0:
        correct_name_pd = names_df[names_df['id'] == correct_spelling_id]
        text_name_pd = pd.concat([text_name_pd, correct_name_pd], axis=0).drop_duplicates()
        correct_name_synonym_id = correct_name_pd['synonym_id'].values[0]
        if correct_name_synonym_id >= 0:
            correct_synonyms = names_df[names_df['synonym_id'] == correct_name_synonym_id]
            text_name_pd = pd.concat([text_name_pd, correct_synonyms], axis=0).drop_duplicates()        
    text_name_pd = expand_dataframe(text_name_pd, names_df)
    text_name_pd = text_name_pd[text_name_pd['deprecated'] == 0]
    if text_name_pd.empty:
        if synonym_id >= 0:
            try:
                text_name = manual_synonyms[str(synonym_id)]
            except Exception:
                log_error(f"Problems with observation and synonym {observation_id}, {synonym_id}")
        elif correct_spelling_id >= 0:
            text_name = names_df[names_df['id'] == correct_spelling_id]['text_name'].values[0]
        else:
            text_name = names_df[names_df['id'] == name_id]['text_name'].values[0]
    else:
        text_name = text_name_pd.sort_values(by=['text_name'], key=lambda x: x.str.len()).iloc[0]['text_name']
    text_name = re.sub(r'[<>:"\|?*]', " ", text_name)

    if text_name.endswith("series") or text_name.endswith("group") or any(x in text_name for x in ["sect.", "subg.", "Mixed"]) or len(text_name.split()) == 1:
        #folder_path = os.path.join(unknown_folder, text_name)
        continue
    else:
        species_base = " ".join(text_name.split()[:2])
        species_folder = os.path.join(known_folder, species_base)
        specific_folder = text_name if any(x in text_name for x in ["var.", "f.", "subsp."]) else f"{text_name} species"
        folder_path = os.path.join(species_folder, specific_folder)
        
    url = f"https://mushroomobserver.org/images/640/{image_id}.jpg"
    file_path = os.path.join(folder_path, f"{image_id}.jpg")

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Save the image content
            with open(file_path, "wb") as img_file:
                img_file.write(response.content)
        elif response.status_code == 403:
            log_error(f"Access forbidden for image {image_id} from {url} (403)")
        elif response.status_code == 404:
            log_error(f"Image {image_id} not found at {url} (404)")
        else:
            log_error(f"Unexpected response {response.status_code} for image {image_id} from {url}")
        save_progress(image_id)  # Salva l'ID come completato

    except requests.exceptions.RequestException as e:
        log_error(f"Error downloading image {image_id} from {url}: {str(e)}")
