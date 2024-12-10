import os
import random
from PIL import Image

def collect_images_recursively(directory):
    """
    Raccoglie tutte le immagini .jpg in modo ricorsivo da una directory e dalle sue sottocartelle.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def mixup_augmentation(directory, output_dir, num_mixup):
    # Raccogli tutte le immagini in modo ricorsivo
    image_paths = collect_images_recursively(directory)
    num_images = len(image_paths)

    # Crea l'output directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(num_mixup):
        # Seleziona due immagini casualmente
        img1_path, img2_path = random.sample(image_paths, 2)
        
        # Carica le immagini
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Converti entrambe le immagini in RGB per assicurare compatibilità
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        # Ottieni dimensioni originali
        w1, h1 = img1.size
        w2, h2 = img2.size

        # Trova il lato maggiore e ridimensiona entrambe le immagini al quadrato
        square_size = max(w1, h1, w2, h2)
        img1_square = img1.resize((square_size, square_size))
        img2_square = img2.resize((square_size, square_size))

        # MixUp delle immagini
        mixup_image = Image.blend(img1_square, img2_square, alpha=0.5)

        # Dimensioni rettangolari finali
        final_width = (w1 + w2) // 2
        final_height = (h1 + h2) // 2
        mixup_image = mixup_image.resize((final_width, final_height))

        # Salva l'immagine MixUp
        output_filename = f"mixup_{idx+1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        mixup_image.save(output_path)

# Percorso della directory principale
directory_path = 'temp_crops'

# Itera sulle sottodirectory
for root, dirs, files in os.walk(directory_path):
    if root == directory_path:
        for folder in dirs:
            folder_path = os.path.join(root, folder)

            # Conta tutte le immagini nella cartella e nelle sottocartelle
            image_paths = collect_images_recursively(folder_path)
            image_count = len(image_paths)
            if image_count < 2:
                print(f"Skipping folder {folder}: not enough images for MixUp.")
                continue

            num_mixup = max(1, round(0.1 * image_count))  # Incremento del 20%

            # Crea una subfolder '{folder} mixup'
            mixup_dir = os.path.join(folder_path, f"{folder} mixup")
            os.makedirs(mixup_dir, exist_ok=True)

            # Applica MixUp augmentation
            mixup_augmentation(folder_path, mixup_dir, num_mixup)