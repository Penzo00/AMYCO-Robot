import os
import shutil
import glob
import random

# Percorso principale della directory originale
main_dir = "temp_crops"

# Nuova struttura dataset
output_dir = "MO13k"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

# Frazione di train/val
train_fraction = 0.8

# Funzione per trasferire dataset con suddivisione train-val
def split_dataset(main_dir, train_dir, val_dir, train_fraction):
    # Crea le directory di output
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Itera su tutte le cartelle principali (folder{i})
    for folder in os.listdir(main_dir):
        folder_path = os.path.join(main_dir, folder)

        # Salta se non è una directory
        if not os.path.isdir(folder_path):
            continue

        # Trova tutte le immagini .jpg nella cartella corrente e nelle sottocartelle
        image_paths = glob.glob(f"{folder_path}/**/*.jpg", recursive=True)

        # Shuffle casuale per distribuzione casuale
        random.shuffle(image_paths)

        # Suddivisione train-val
        train_size = int(len(image_paths) * train_fraction)
        train_images = image_paths[:train_size]
        val_images = image_paths[train_size:]

        # Percorsi di output per questa cartella
        train_folder = os.path.join(train_dir, folder)
        val_folder = os.path.join(val_dir, folder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # Controlla se entrambe le cartelle di destinazione hanno immagini .jpg
        train_images_exist = glob.glob(f"{train_folder}/*.jpg")
        val_images_exist = glob.glob(f"{val_folder}/*.jpg")

        if train_images_exist and val_images_exist:
            # Elimina la cartella di origine
            shutil.rmtree(folder_path)
            print(f"Cartella eliminata: {folder_path}")
            continue

        # Trasferisci le immagini in train
        for img_path in train_images:
            dest_path = os.path.join(train_folder, os.path.basename(img_path))
            shutil.move(img_path, dest_path)

        # Trasferisci le immagini in val
        for img_path in val_images:
            dest_path = os.path.join(val_folder, os.path.basename(img_path))
            shutil.move(img_path, dest_path)

        # Controlla se entrambe le cartelle di destinazione hanno immagini .jpg
        train_images_exist = glob.glob(f"{train_folder}/*.jpg")
        val_images_exist = glob.glob(f"{val_folder}/*.jpg")

        if train_images_exist and val_images_exist:
            # Elimina la cartella di origine
            shutil.rmtree(folder_path)
            print(f"Cartella eliminata: {folder_path}")

        print(f"Completato split per {folder}: {len(train_images)} train, {len(val_images)} val.")

# Esegui lo split
split_dataset(main_dir, train_dir, val_dir, train_fraction)
