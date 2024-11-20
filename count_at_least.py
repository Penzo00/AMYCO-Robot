import os

# Imposta il percorso della directory principale
main_dir = "MO2/classes"

# Definisci le estensioni delle immagini da considerare
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Variabile per contare le sottocartelle principali con almeno 150 immagini complessive
count_main_folders_with_150_or_more = 0

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

# Itera su ogni sottocartella principale di "Known species"
for main_subfolder in os.listdir(main_dir):
    main_subfolder_path = os.path.join(main_dir, main_subfolder)
    if os.path.isdir(main_subfolder_path):
        # Conta tutte le immagini nelle sottocartelle foglia di questa sottocartella principale
        image_count = count_images_in_leaf_subfolders(main_subfolder_path)
        # Verifica se il conteggio raggiunge o supera 150
        if image_count == 44:
            count_main_folders_with_150_or_more += 1
            print(f"{main_subfolder}")

print(f"Numero di sottocartelle principali con almeno 42 immagini complessive nelle sottocartelle foglia: {count_main_folders_with_150_or_more}")
