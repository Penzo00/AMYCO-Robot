import os
import pandas as pd
from PIL import Image
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(
    filename='verification_errors.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def sanitize_folder_name(name):
    """
    Replace or remove characters that are invalid in Windows folder names.
    """
    invalid_chars = {
        '<': '',
        '>': '',
        ':': '',
        '"': '',
        '/': '',
        '\\': '',
        '|': '',
        '?': '',
        '*': ''
    }
    for char, replacement in invalid_chars.items():
        name = name.replace(char, replacement)
    return name.strip()


def determine_folder(text_name):
    """
    Determine the appropriate folder path for the image based on the text_name.
    """
    if (text_name.lower().endswith('series') or
            text_name.lower().endswith('group') or
            'sect.' in text_name or
            len(text_name.split()) == 1):
        folder_type = 'Unknown species'
        folder_name = 'Unknown species'
    else:
        folder_type = 'Known species'
        parts = text_name.split()
        if len(parts) >= 2:
            first_two = ' '.join(parts[:2])
        else:
            first_two = text_name
        if not any(x in text_name for x in ['var.', 'f.', 'subsp.']):
            folder_full_name = f"{text_name} species"
        else:
            folder_full_name = text_name
        folder_name = sanitize_folder_name(folder_full_name)
        first_two = sanitize_folder_name(first_two)
        folder_name = os.path.join(folder_type, first_two, folder_name)
    return folder_name


def replace_unix_characters(name):
    """
    Replace Unix-specific characters with Windows-compatible ones.
    """
    return sanitize_folder_name(name)


def resize_image_if_needed(file_path):
    """
    Resize the image so that the smaller dimension is exactly 960 pixels,
    adjusting the larger dimension to maintain the aspect ratio.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            min_dim = min(width, height)
            if min_dim > 960:
                scale_factor = 960 / min_dim
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.save(file_path, format=img.format)
                print(f"Resized {file_path.name} to: {new_width}x{new_height}")
                return True
        return False
    except Exception as e:
        logging.error(f"Error resizing image {file_path.name}: {e}")
        return False


def verify_image_path_and_resize(image_id, csv_mappings, images_dir):
    """
    Verify the correct folder for an image and resize it if needed.
    """
    # Retrieve the necessary mappings from the CSV
    observation_id = csv_mappings['image_to_observation'].get(str(image_id))
    if not observation_id:
        logging.error(f"Observation ID not found for image ID {image_id}")
        return False

    name_id = csv_mappings['observation_to_name'].get(str(observation_id))
    if not name_id:
        logging.error(f"Name ID not found for observation ID {observation_id}")
        return False

    synonym_id = csv_mappings['name_to_synonym'].get(str(name_id))
    if pd.isna(synonym_id) or str(synonym_id).upper() == 'NULL' or str(synonym_id).strip() == '':
        text_name = csv_mappings['name_to_text_name'].get(str(name_id))
    else:
        synonym_texts = csv_mappings['synonym_to_texts'].get(str(synonym_id), [])
        if not synonym_texts:
            logging.error(f"No text names found for synonym ID {synonym_id}")
            return False
        text_name = min(synonym_texts, key=len)

    # Determine the correct folder
    correct_folder = Path(images_dir) / determine_folder(text_name)

    # Check if the image is already in the correct folder
    found = False
    for ext in ['.jpg', '.jpeg', '.png']:
        file_path = correct_folder / f"{image_id}{ext}"
        if file_path.exists():
            found = True
            resize_image_if_needed(file_path)
            break

    # If not found, search the image in the wrong folder and move it
    if not found:
        print(f"Image {image_id} not found in correct folder, searching...")
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.startswith(str(image_id)):
                    old_path = Path(root) / file
                    new_path = correct_folder / file
                    correct_folder.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(new_path))
                    print(f"Moved {old_path} to {new_path}")
                    resize_image_if_needed(new_path)
                    return True
    return False


def load_csv_mappings(csv_dir):
    """
    Load all necessary mappings from the CSV files.
    """
    try:
        # Load CSVs
        names_df = pd.read_csv(Path(csv_dir) / "names.csv", sep='\t', dtype=str)
        observations_df = pd.read_csv(Path(csv_dir) / "observations.csv", sep='\t', dtype=str)
        images_observations_df = pd.read_csv(Path(csv_dir) / "images_observations.csv", sep='\t', dtype=str)

        # Create mappings
        image_to_observation = images_observations_df.set_index('image_id')['observation_id'].to_dict()
        observation_to_name = observations_df.set_index('id')['name_id'].to_dict()
        name_to_synonym = names_df.set_index('id')['synonym_id'].to_dict()
        name_to_text_name = names_df.set_index('id')['text_name'].to_dict()
        synonym_to_texts = names_df[names_df['deprecated'] == '0'].dropna(subset=['synonym_id']).groupby('synonym_id')[
            'text_name'].apply(list).to_dict()

        # Return all mappings as a dictionary
        return {
            'image_to_observation': image_to_observation,
            'observation_to_name': observation_to_name,
            'name_to_synonym': name_to_synonym,
            'name_to_text_name': name_to_text_name,
            'synonym_to_texts': synonym_to_texts,
        }
    except Exception as e:
        logging.error(f"Error loading CSV files or creating mappings: {e}")
        return None


def verify_and_correct_all_images(csv_dir, images_dir):
    """
    Verify and correct the location and size of all images in the 'MO' folder.
    """
    csv_mappings = load_csv_mappings(csv_dir)
    if csv_mappings is None:
        print("Failed to load CSV mappings. Check the logs for errors.")
        return

    # Iterate over each image in the 'MO' folder
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_id = file.split('.')[0]  # Get the image ID from the file name
                verify_image_path_and_resize(image_id, csv_mappings, images_dir)


if __name__ == "__main__":
    # Set the paths to the CSV files directory and the images directory
    csv_dir = "CSV"  # Update this if your CSV files are stored elsewhere
    images_dir = "MO"  # Update this if your images are stored elsewhere

    # Verify and correct the images
    verify_and_correct_all_images(csv_dir, images_dir)