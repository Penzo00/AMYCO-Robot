import os
import shutil

# Imposta il percorso della directory principale
main_dir = "MO2/classes"

# Definisci le estensioni delle immagini da considerare
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Funzione per contare le immagini in tutte le "sottocartelle_delle_sottocartelle" di una sottocartella principale
def count_images_in_leaf_subfolders(directory):
    total_images = 0
    # Itera solo sulle sottocartelle di `directory`, cioè le `sottocartelle_delle_sottocartelle`
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            # Conta le immagini nella sottocartella foglia
            for _, _, files in os.walk(subfolder_path):
                total_images += sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
    return total_images

# Itera su ogni sottocartella principale di "MO2/classes"
for main_subfolder in os.listdir(main_dir):
    main_subfolder_path = os.path.join(main_dir, main_subfolder)
    if os.path.isdir(main_subfolder_path):
        # Conta tutte le immagini nelle sottocartelle foglia di questa sottocartella principale
        image_count = count_images_in_leaf_subfolders(main_subfolder_path)
        
        # Se il conteggio delle immagini è inferiore a 44, elimina la cartella principale
        if image_count < 42:
            shutil.rmtree(main_subfolder_path)  # Elimina la cartella e tutto il suo contenuto

print("Eliminazione completata delle cartelle con meno di 42 immagini.")
