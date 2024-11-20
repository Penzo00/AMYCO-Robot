import os
import matplotlib.pyplot as plt
import pandas as pd

# Imposta il percorso della directory principale
main_dir = "MO2/classes"

# Definisci le estensioni delle immagini da considerare
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Crea una lista per salvare il numero di immagini per ciascuna sottocartella foglia
image_counts = []

# Funzione per verificare se una cartella è una "sottocartella foglia" (non ha altre sottocartelle)
def is_leaf_directory(directory):
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            return False
    return True

# Naviga attraverso tutte le sottocartelle e trova quelle "foglia"
for root, dirs, files in os.walk(main_dir):
    # Se la cartella è una "foglia", contiamo le immagini
    if is_leaf_directory(root):
        image_count = sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
        image_counts.append(image_count)

# Analisi dei dati
# Creiamo categorie per contare le sottocartelle per fasce di immagini
bins = [0, 10, 20, 30, 50, 150, 300, 500, 1000, 5000]
labels = ["<10", "10-20", "20-30", "30-50", "50-150", "150-300", "300-500", "500-1000", ">1000"]
image_counts_binned = pd.cut(image_counts, bins=bins, labels=labels, right=False)

# Contiamo quante sottocartelle rientrano in ogni fascia
distribution = image_counts_binned.value_counts().sort_index()

# Visualizza e salva l'istogramma
plt.figure(figsize=(10, 6))
distribution.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Distribution of the number of images per species")
plt.xlabel("Number of images")
plt.ylabel("Number of species")
plt.xticks(rotation=45)

# Salva l'istogramma
plt.savefig("distribuzione_sottocartelle_foglia_YOLOn.png", format="png", dpi=300)