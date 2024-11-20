import os
from PIL import Image
import numpy as np

# Imposta il percorso della directory principale
main_dir = "MO2/classes"

# Definisci le estensioni delle immagini da considerare
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Liste per memorizzare larghezza e altezza di ogni immagine
widths = []
heights = []

# Funzione per ottenere le dimensioni di tutte le immagini nelle sottocartelle foglia
def get_image_sizes(directory):
    for root, dirs, files in os.walk(directory):
        # Controlla se è una sottocartella foglia (senza sottocartelle)
        if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_path = os.path.join(root, file)
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
median_width = int(np.median(widths))
median_height = int(np.median(heights))

print(f"La dimensione mediana delle immagini è: {median_width}x{median_height}")
