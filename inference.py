import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Carica il modello YOLO
print("Loading model...")
model = YOLO("runs/detect/train/weights/best.pt")

# Definisci le cartelle di input e output
input_dir = Path("Known species")
output_images_dir = Path("train/images")
output_labels_dir = Path("train/labels")

# Crea le cartelle di output se non esistono
output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

# Colleziona tutti i percorsi delle immagini nelle sottocartelle
image_paths = list(input_dir.rglob("*.jpg"))

print("Inference...")

# Esegui inferenza solo sui file che non sono già processati
for image_path in image_paths:
    output_image_path = output_images_dir / image_path.name
    label_file_path = output_labels_dir / f"{image_path.stem}.txt"

    # Salta l'immagine se è già processata
    if output_image_path.exists() and label_file_path.exists():
        continue

    # Verifica che il file sia un'immagine leggibile
    try:
        img = Image.open(image_path)
        img.verify()  # Verifica che il file sia un'immagine valida
    except (IOError, SyntaxError) as e:
        print(f"File non valido o corrotto: {image_path}, errore: {e}")
        continue  # Salta il file e passa al successivo

    # Esegui inferenza sull'immagine
    results = model(image_path, conf=0.8, verbose=False)  # Singola immagine, senza stream

    # Controlla se ci sono box nei risultati per l'immagine corrente
    if len(results[0].boxes) > 0:
        # Copia l'immagine con rilevamenti nella cartella di output
        shutil.copy(image_path, output_image_path)

        # Salva le coordinate normalizzate in un file .txt
        with open(label_file_path, "w") as f:
            for box in results[0].boxes:
                # Estrai la classe e le coordinate normalizzate xywhn
                cls = int(box.cls.item())  # Classe come intero
                x_center, y_center, width, height = box.xywhn[0].tolist()

                # Scrivi la classe e le coordinate nel file .txt nel formato YOLO
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

print("Inference complete.")