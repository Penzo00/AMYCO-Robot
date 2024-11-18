import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import subprocess
import pandas as pd
import os
import re

# Load CSV files
print("Loading CSV files...")
names_df = pd.read_csv("CSV/names.csv", sep="\t").drop(['author', 'rank'], axis=1).fillna(-1)
names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']] = names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']].astype('int32')
descriptions = pd.read_csv('name_descriptions.csv', delimiter='\t', on_bad_lines='skip').fillna("")

print("Loading models...")
# Carica il modello YOLO11
model = YOLO("best_ncnn_model")
class_names = np.load('classes.npy')
# Carica il modello ONNX
ort_session = ort.InferenceSession("tl_model.onnx")

# Funzione per eseguire l'inferenza con YOLO11 e ritagliare il box con la probabilità maggiore
def infer_and_crop():
    # Inizializza la webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Leggi un frame dalla webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Esegui l'inferenza con YOLO11 sul frame
        results = model(source=frame, stream=True)

        # Trova il box con la probabilità maggiore
        best_box = None
        if results.boxes:
            best_box = results.boxes[results.boxes == max(results.probs)]

        if best_box is not None:
            # Ritaglia il box con la probabilità maggiore
            x1, y1, x2, y2 = map(int, best_box.xyxy)
            cropped_image = frame[y1:y2, x1:x2]

            # Ridimensiona l'immagine a 309x298
            image_resized = cv2.resize(cropped_image, (298, 309))

            # Converti l'immagine in float32 e normalizza i valori
            input_data = image_resized.astype(np.float32) / 255.0

            # Aggiungi una dimensione batch
            input_data = np.expand_dims(input_data, axis=0)

            # Ottieni il nome dell'input del modello ONNX
            input_name = ort_session.get_inputs()[0].name

            # Esegui l'inferenza con il modello ONNX
            results = ort_session.run(None, {input_name: input_data})

            # Ottieni le probabilità delle classi
            probabilities = results[0][0]

            # Ordina le probabilità e ottieni gli indici delle prime 5 classi
            top_5_indices = np.argsort(probabilities)[-5:][::-1]

            # Prepara il testo da far pronunciare a Piper
            text_to_speak = "Top 5 classes: "
            for i in top_5_indices:
                name_id = names_df[names_df["text_name"] == class_names[i]]["id"]
                text_to_speak += f"Class {class_names[i]}, Probability {probabilities[i]:.4f}." + f"{tell_description(name_id)}"

            # Usa Piper per far pronunciare i risultati dell'inferenza
            subprocess.run(
                f"echo '{text_to_speak}' | ./piper --model it_IT-paola-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -",
                shell=True,
                check=True,
            )

        # Mostra il frame con il box ritagliato (opzionale)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def tell_description(name_id):
    def pulisci_testo(testo):
        # Sostituisce i caratteri \n con uno spazio, poi rimuove link, caratteri _ e []
        return re.sub(r'\s+', ' ', re.sub(r'[_\[\]]', '', re.sub(r'http\S+|www\S+', '', testo.replace('\\n', ' ')))).strip()
    
    # Trova l'indice della riga con il valore specifico di name_id (come stringa)
    start_index = descriptions.index[descriptions['name_id'] == name_id].tolist()
    
    if not start_index:
        print(f"Nessuna descrizione aggiuntiva per {class_names[i]}")
        return
    start_index = start_index[0]
    
    # Lista per accumulare i valori da concatenare
    testo_concatenato = []

    # Inizia dalla riga successiva alla riga trovata
    for i in range(start_index, len(descriptions)):
        # Prendi i valori delle celle `id` e `name_id` della riga successiva
        next_id_val = descriptions.at[i + 1, 'id'] if i + 1 < len(df) else None
        next_name_id_val = descriptions.at[i + 1, 'name_id'] if i + 1 < len(df) else None
        
        # Interrompi se `id` o `name_id` della riga successiva contengono un numero intero come stringa
        if (isinstance(next_id_val, str) and next_id_val.isdigit()) or \
           (isinstance(next_name_id_val, str) and next_name_id_val.isdigit()):
            break

        # Aggiungi alla lista i valori della riga corrente, escluso `name_id`
        testo_concatenato.extend(map(str, descriptions.iloc[i].drop('name_id').values))
    
    # Unisci i testi e puliscili
    testo_finale = pulisci_testo(' '.join(testo_concatenato))
    return testo_finale[3:]