import os
import pandas as pd
import requests
from tqdm import tqdm
import logging
from pathlib import Path
import re
from PIL import Image

# Configure logging
logging.basicConfig(filename='errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def sanitize_folder_name(name):
    """
    Replace or remove characters that are invalid in Windows folder names.
    """
    # Define a mapping of invalid characters to their replacements or remove them
    invalid_chars = {
        '<': '',
        '>': '',
        ':': '',
        '"': '',
        '\\': '',
        '|': '',
        '?': '',
        '*': ''
    }
    # Replace invalid characters
    for char, replacement in invalid_chars.items():
        name = name.replace(char, replacement)
    # Additionally, strip leading/trailing whitespace
    return name.strip()


def determine_folder(text_name):
    """
    Determine the appropriate folder path for the image based on the text_name.
    """
    # Check for Unknown species criteria
    if (text_name.lower().endswith('series') or
            text_name.lower().endswith('group') or
            'sect.' in text_name or
            len(text_name.split()) == 1):
        folder_name = os.path.join('Unknown species', text_name)
    else:
        folder_type = 'Known species'
        # Take the first two parts of the name
        parts = text_name.split()
        if len(parts) >= 2:
            first_two = ' '.join(parts[:2])
        else:
            first_two = text_name
        # Check for "var.", "f.", or "subsp."
        if not any(x in text_name for x in ['var.', 'f.', 'subsp.']):
            folder_full_name = f"{text_name} species"
        else:
            folder_full_name = text_name
        folder_name = sanitize_folder_name(folder_full_name)
        # Further sanitize first_two
        first_two = sanitize_folder_name(first_two)
        folder_name = os.path.join(folder_type, first_two, folder_name)
    return folder_name


def replace_unix_characters(name):
    """
    Replace Unix-specific characters with Windows-compatible ones.
    """
    # This function can be expanded based on specific requirements
    return sanitize_folder_name(name)


def download_image(image_id, dest_path, session):
    """
    Attempt to download an image with various extensions. If successful, save it to dest_path.
    """
    extensions = ['.jpg', '.jpeg', '.png']
    for ext in extensions:
        url = f"https://storage.googleapis.com/mo-image-archive-bucket/orig/{image_id}{ext}"
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    f.write(response.content)
                return True
        except requests.RequestException:
            continue
    return False


def resize_image(file_path):
    """
    Resize the image so that the smaller dimension is exactly 960 pixels,
    adjusting the larger dimension to maintain the aspect ratio.
    If the smaller dimension is already 960 pixels or less, the image is left unchanged.

    Parameters:
        file_path (Path): The path to the image file to be resized.

    Returns:
        bool: True if resizing was successful or not needed, False if an error occurred.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            # Determine the smaller dimension
            min_dim = min(width, height)
            if min_dim > 960:
                # Calculate the scaling factor
                scale_factor = 960 / min_dim
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # Resize the image with high-quality downsampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                # Save the resized image, preserving the original format
                img.save(file_path)
        return True
    except Exception as e:
        logging.error(f"Error resizing image {file_path.name}: {e}")
        return False

def read_progress(progress_file):
    """
    Read the last processed image_id from the progress file.

    Parameters:
        progress_file (Path): Path to the progress tracking file.

    Returns:
        int: The last processed image_id. Returns 0 if file does not exist.
    """
    if not progress_file.exists():
        return 0
    try:
        with open(progress_file, 'r') as f:
            last_id = f.read().strip()
            return int(last_id) if last_id.isdigit() else 0
    except Exception as e:
        logging.error(f"Error reading progress file: {e}")
        return 0


def update_progress(progress_file, image_id):
    """
    Update the progress file with the latest processed image_id.

    Parameters:
        progress_file (Path): Path to the progress tracking file.
        image_id (int): The image_id that was just processed.
    """
    try:
        with open(progress_file, 'w') as f:
            f.write(str(image_id))
    except Exception as e:
        logging.error(f"Error updating progress file: {e}")


def main():
    """
    Main function to orchestrate the downloading of images based on CSV data.
    Utilizes a progress file to track and resume progress.
    """
    # Define paths
    csv_dir = Path("CSV")
    images_dir = Path("MO")
    progress_file = Path("progress.txt")

    # Read CSV files
    print("Loading CSV files...")
    try:
        names_df = pd.read_csv(csv_dir / "names.csv", sep='\t', dtype=str)
        observations_df = pd.read_csv(csv_dir / "observations.csv", sep='\t', dtype=str)
        images_observations_df = pd.read_csv(csv_dir / "images_observations.csv", sep='\t', dtype=str)
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        return

    # Create necessary mappings
    print("Creating mappings...")
    try:
        # Mapping from observation_id to name_id
        observation_to_name = observations_df.set_index('id')['name_id'].to_dict()

        # Mapping from name_id to synonym_id
        name_to_synonym = names_df.set_index('id')['synonym_id'].to_dict()

        # Mapping from synonym_id to list of text_names where deprecated == 0
        # Exclude rows where synonym_id is NULL
        synonym_group = names_df[names_df['deprecated'] == '0'].dropna(subset=['synonym_id']).groupby('synonym_id')['text_name'].apply(list).to_dict()
    except Exception as e:
        logging.error(f"Error creating mappings: {e}")
        return

    # Prepare list of image_id and corresponding observation_id
    try:
        images_observations = images_observations_df[['image_id', 'observation_id']].drop_duplicates()
        # Ensure image_id is integer for sorting
        images_observations['image_id'] = images_observations['image_id'].astype(int)
    except Exception as e:
        logging.error(f"Error processing images_observations.csv: {e}")
        return

    # Initialize a requests session for connection pooling
    session = requests.Session()

    # Read the last processed image_id
    last_processed_id = read_progress(progress_file)
    print(f"Last processed image_id: {last_processed_id}")

    # Sort image_ids in ascending order
    images_observations_sorted = images_observations.sort_values('image_id')

    # Calculate total images and already processed images
    total_images_total = len(images_observations_sorted)
    images_already_processed = len(images_observations_sorted[images_observations_sorted['image_id'] <= last_processed_id])

    # Filter images to process
    images_to_process = images_observations_sorted[images_observations_sorted['image_id'] > last_processed_id]
    total_images_remaining = len(images_to_process)

    if total_images_remaining == 0:
        print("All images have been processed.")
        return

    # Initialize tqdm with total and initial
    pbar = tqdm(total=total_images_total, initial=images_already_processed, desc="Downloading and resizing images",
                unit="image")

    # Iterate over each image
    print("Starting image downloads...")
    for _, row in images_to_process.iterrows():
        image_id = row['image_id']
        observation_id = row['observation_id']

        # Determine the folder name
        name_id = observation_to_name.get(str(observation_id))
        if not name_id:
            logging.error(f"Name ID not found for observation ID {observation_id}")
            pbar.update(1)
            continue

        synonym_id = name_to_synonym.get(str(name_id))
        text_names = []

        if pd.isna(synonym_id) or str(synonym_id).upper() == 'NULL' or str(synonym_id).strip() == '':
            # If synonym_id is NULL, use the text_name directly
            text_name_row = names_df[names_df['id'] == name_id]
            if text_name_row.empty:
                logging.error(f"text_name not found for name_id {name_id}")
                pbar.update(1)
                continue
            selected_name = text_name_row.iloc[0]['text_name']
        else:
            # Use synonym_id to get the group of text_names
            text_names = synonym_group.get(str(synonym_id), [])
            if not text_names:
                logging.error(f"No text names found for synonym ID {synonym_id}")
                pbar.update(1)
                continue
            # Select the shortest text_name
            selected_name = min(text_names, key=len)

        # Determine folder path
        folder_path = determine_folder(selected_name)
        full_folder_path = images_dir / folder_path
        # Replace Unix characters
        full_folder_path = Path(replace_unix_characters(str(full_folder_path)))
        # Create directories if they don't exist
        full_folder_path.mkdir(parents=True, exist_ok=True)
        # Define possible file paths
        possible_extensions = ['.jpg', '.jpeg', '.png']
        # Check if image already exists
        image_exists = False
        existing_extension = None
        for ext in possible_extensions:
            file_path = full_folder_path / f"{image_id}{ext}"
            if file_path.exists():
                image_exists = True
                existing_extension = ext
                break
        if image_exists:
            # Update progress file since this image is already processed
            update_progress(progress_file, image_id)
            pbar.update(1)
            continue  # Skip downloading

        # Attempt to download the image
        success = False
        for ext in possible_extensions:
            dest_path = full_folder_path / f"{image_id}{ext}"
            if download_image(str(image_id), dest_path, session):
                if download_image(str(image_id), dest_path, session):
                    # After successful download, attempt to resize the image
                    if resize_image(dest_path):
                        success = True
                    else:
                        logging.error(f"Resizing failed for image {image_id}.")
                        # Optionally, you can choose to delete the corrupted image
                        # dest_path.unlink(missing_ok=True)
                    break  # Stop trying other extensions if successful
        if success:
            # Update progress file upon successful download
            update_progress(progress_file, image_id)
        else:
            logging.error(f"Failed to download and resize image {image_id} with all extensions.")

        # Update the progress bar
        pbar.update(1)

    pbar.close()
    print("Image downloading and resizing completed.")

if __name__ == "__main__":
    main()