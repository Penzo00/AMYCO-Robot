import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import random
import shutil

# Percorsi e configurazione
source_dir = Path("fastervit2_training/tmp3")
output_base_dir = Path("fastervit2_training")
output_crops_dir = output_base_dir / "temp_crops"

# Carica il modello YOLO
model = YOLO("best_ncnn_model2", task="detect")

# Passo 1: Inferenza e ritaglio
def infer_and_crop_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = Path(root) / file

                # Esegui inferenza sull'immagine
                try:
                    img = Image.open(image_path)
                    img.verify()
                    img = Image.open(image_path)
                except (IOError, SyntaxError) as e:
                    print(f"File non valido o corrotto: {image_path}, errore: {e}")
                    continue

                results = model(img, conf=0.5713947415351868, verbose=False)

                # Se ci sono bounding box, ritaglia e salva
                if len(results[0].boxes) > 0:
                    # Percorso relativo per mantenere la struttura originale
                    relative_path = image_path.relative_to(input_dir)
                    subfolder_path = relative_path.parent

                    # Crea la directory di output con la struttura originale
                    crop_output_dir = output_dir / subfolder_path
                    crop_output_dir.mkdir(parents=True, exist_ok=True)

                    for i, box in enumerate(results[0].boxes):
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        cropped_img = img.crop((x_min, y_min, x_max, y_max))

                        # Salva il ritaglio
                        crop_image_path = crop_output_dir / f"{image_path.stem}_crop_{i}.jpg"
                        cropped_img.save(crop_image_path)

infer_and_crop_images(source_dir, output_crops_dir)