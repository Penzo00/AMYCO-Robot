import os

def convert_segmentation_to_bbox(file_path):
    # Carica i dati dal file di testo
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    converted_labels = []
    
    for line in lines:
        # Estrai i valori come float
        values = list(map(float, line.strip().split()))
        class_id = 0  # Imposta sempre la classe a 0 come desiderato
        
        # Ottieni le coordinate X e Y dei punti
        x_coords = values[1::2]
        y_coords = values[2::2]
        
        # Trova i minimi e i massimi per creare il bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calcola le dimensioni del bounding box nel formato YOLO
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Formatta la riga nel formato YOLO
        converted_label = f"{class_id} {x_center} {y_center} {width} {height}\n"
        converted_labels.append(converted_label)
    
    # Sovrascrivi il file originale con i dati convertiti
    with open(file_path, 'w') as f:
        f.writelines(converted_labels)
    
    print(f"Etichette convertite sovrascritte in: {file_path}")

def process_main_folder(main_folder):
    # Definisci le sottocartelle da esplorare
    subfolders = ["test", "train", "valid"]
    
    for subfolder in subfolders:
        # Costruisci il percorso della cartella "labels" in ciascuna sottocartella
        labels_folder = os.path.join(main_folder, subfolder, "labels")
        
        # Verifica se la cartella "labels" esiste
        if os.path.isdir(labels_folder):
            print(f"Processando i file nella cartella: {labels_folder}")
            # Itera su tutti i file .txt nella cartella "labels"
            for file_name in os.listdir(labels_folder):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(labels_folder, file_name)
                    convert_segmentation_to_bbox(file_path)
        else:
            print(f"La cartella 'labels' non esiste in: {os.path.join(main_folder, subfolder)}")

# Esempio di utilizzo
main_folder = '.'  # Usa '.' se il codice è eseguito nella directory principale contenente le cartelle 'test', 'train', e 'valid'
process_main_folder(main_folder)