import glob
from PIL import Image
import numpy as np
import os

# Imposta il percorso della directory principale
main_dir = "MO13k"

# Definisci le estensioni delle immagini da considerare
image_extensions = {".jpg"}

# Liste per memorizzare larghezza e altezza di ogni immagine
widths = []
heights = []

# Funzione per ottenere le dimensioni di tutte le immagini nelle sottocartelle
def get_image_sizes(directory):
    # Trova tutti i file .jpg nelle sottocartelle
    image_paths = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive=True)
    for image_path in image_paths:
        try:
            # Apri l'immagine e ottieni le dimensioni
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except Exception as e:
            print(f"Impossibile aprire l'immagine {image_path}: {e}")

# Esegui la funzione sulla directory principale
get_image_sizes(main_dir)

# Calcola le mediane
if widths and heights:  # Assicurati che ci siano immagini valide
    median_width = int(np.median(widths))
    median_height = int(np.median(heights))
    print(f"La dimensione mediana delle immagini è: {median_width}x{median_height}")
else:
    print("Nessuna immagine valida trovata.")