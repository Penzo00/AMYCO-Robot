import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Carica il modello YOLO
model = YOLO("runs/detect/train17/weights/best.pt")

# Definisci le cartelle di output
output_crops_dir = Path("MO2/classes")  # Cartella per salvare i ritagli dei box
output_labels_dir = Path("MO2/training/labels")

# Crea le cartelle di output per le etichette se non esistono
output_labels_dir.mkdir(parents=True, exist_ok=True)

# Percorso della cartella di input contenente le immagini
input_dir = Path("Known species")

# Usa os.walk per scorrere tutte le sottocartelle e cercare immagini
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".jpg"):  # Verifica se è un file jpg
            image_path = Path(root) / file

            # Esegui inferenza
            try:
                img = Image.open(image_path)
                img.verify()  # Verifica che il file sia un'immagine valida
                img = Image.open(image_path)  # Riapri l'immagine per ritagliarla (necessario dopo .verify)
            except (IOError, SyntaxError) as e:
                print(f"File non valido o corrotto: {image_path}, errore: {e}")
                continue  # Salta il file e passa al successivo

            results = model(img, conf=0.81, verbose=False)

            # Verifica se ci sono bounding box nei risultati
            if len(results[0].boxes) > 0:
                # Estrai il percorso relativo a partire da "Known species"
                relative_path = image_path.relative_to(input_dir)

                # Estrai la struttura "{subfolder1}/{subfolder2}" dal percorso relativo
                subfolder_path = relative_path.parent  # Ottiene le sottocartelle senza il nome del file

                # Crea la cartella di output per i ritagli con la stessa struttura
                crop_output_dir = output_crops_dir / subfolder_path
                crop_output_dir.mkdir(parents=True, exist_ok=True)

                # Salva le coordinate normalizzate in un file .txt
                label_file_path = output_labels_dir / f"{image_path.stem}.txt"
                with open(label_file_path, "w") as f:
                    for i, box in enumerate(results[0].boxes):
                        # Ritaglia la porzione di immagine all'interno del bounding box (usando xyxy)
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        cropped_img = img.crop((x_min, y_min, x_max, y_max))

                        # Salva il ritaglio con il nome originale e un indice
                        crop_image_path = crop_output_dir / f"{image_path.stem}_crop_{i}.jpg"
                        cropped_img.save(crop_image_path)

                        # Estrai la classe e le coordinate normalizzate (xywhn)
                        cls = int(box.cls.item())
                        x_center, y_center, width, height = box.xywhn[0].tolist()

                        # Scrivi la classe e le coordinate nel file delle etichette nel formato YOLO
                        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

print("Processo completato.")