import os
import hashlib
from pathlib import Path

def file_hash(file_path):
    """Compute MD5 hash of the given file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicates(base_path):
    """Remove duplicate images in the given directory and its subdirectories, along with their corresponding labels."""
    unique_hashes = set()
    
    # Lista di cartelle da analizzare (test, train, valid)
    subfolders = ['test', 'valid', 'train']
    
    for subfolder in subfolders:
        images_path = Path(base_path) / subfolder / 'images'
        labels_path = Path(base_path) / subfolder / 'labels'
        
        # Usa liste per combinare i file .jpeg e .jpg
        image_files = list(images_path.glob('*.jpeg')) + list(images_path.glob('*.jpg'))
        
        for image_file in image_files:
            file_hash_value = file_hash(image_file)
            
            if file_hash_value in unique_hashes:
                # Se il file è duplicato, rimuovilo
                print(f"Removing duplicate image: {image_file}")
                os.remove(image_file)
                
                # Rimuovi la label corrispondente
                label_file = labels_path / f"{image_file.stem}.txt"
                if label_file.exists():
                    print(f"Removing corresponding label: {label_file}")
                    os.remove(label_file)
            else:
                # Aggiungi l'hash del file unico
                unique_hashes.add(file_hash_value)
    print("Done removing duplicate images")

# Esegui la funzione sulla cartella base
remove_duplicates('/detection_training')