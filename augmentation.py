# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
os.environ["KERAS_BACKEND"] = "tensorflow"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)
from glob import glob

import random
random.seed(seed)

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import mixed_precision

print(tf.__version__)

# Conta le immagini per classe
class_counts = {}

for root, dirs, files in os.walk(data_dir):
    if root == data_dir:
        for dir_name in dirs:
            class_path = os.path.join(root, dir_name)
            total_files = 0
            for sub_root, sub_dirs, sub_files in os.walk(class_path):
                total_files += len([file for file in sub_files if file.endswith('.jpg')])
            class_counts[dir_name] = total_files

data_dir = "MO2/classes"
batch_size = 16
square_image_size = (309, 309)
final_image_size = (309, 298)
min_images = 861
# Seleziona le classi con meno di 861 immagini
target_classes = [cls for cls, count in class_counts.items() if count < min_images]
# Funzione per ruotare un'immagine casualmente (da definire)
def random_rotate(image):
    # Esempio: rotazione casuale di 0, 90, 180 o 270 gradi
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return tf.image.rot90(image, k=angle // 90)

# Funzione per mixup (da definire)
def mixup(image1, image2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return lam * image1 + (1 - lam) * image2

# Applica mixup augmentation alle classi selezionate
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    augmented_class_path = os.path.join(augmented_dir, class_name)
    
    # Controlla se la cartella contiene già elementi
    if os.path.exists(augmented_class_path) and len(os.listdir(augmented_class_path)) > 0:
        print(f"Skipping class {class_name} as it already contains augmented images.")
        continue
    
    if not os.path.exists(augmented_class_path):
        os.makedirs(augmented_class_path)
    
    # Prendi tutte le immagini della classe corrente
    all_images_paths = glob(os.path.join(class_path, '**', '*.jpg'), recursive=True)
    all_images = [tf.keras.preprocessing.image.load_img(img_path) for img_path in all_images_paths]
    all_images = [tf.keras.preprocessing.image.img_to_array(img) for img in all_images]
    all_images = [tf.image.resize(img, square_image_size) for img in all_images]
    
    all_images = np.array(all_images)
    current_count = len(all_images)
    print(f"Processing class: {class_name}, current count: {current_count}")
    
    # Augmenta le immagini della classe corrente fino a raggiungere il minimo richiesto
    while current_count < min_images:
        for i in range(len(all_images)):
            # Applica mixup con tutte le immagini successive nella lista
            for j in range(i + 1, len(all_images)):
                # Applica trasformazioni casuali alle immagini di input
                image1 = random_rotate(all_images[i])
                image2 = random_rotate(all_images[j])
                
                # Genera immagine mixup
                mixed_image = mixup(image1, image2)
                
                # Ridimensiona l'immagine augmentata a 309x298
                mixed_image_resized = tf.image.resize(mixed_image, final_image_size)
                
                # Salva l'immagine augmentata e ridimensionata
                image_path = os.path.join(augmented_class_path, f'aug_{current_count + 1}.jpg')
                tf.keras.preprocessing.image.save_img(image_path, mixed_image_resized.numpy())
                current_count += 1
                
                # Termina il ciclo se raggiungi il numero minimo di immagini
                if current_count >= min_images:
                    break
            if current_count >= min_images:
                break
        if current_count >= min_images:
            break
    
    print(f"Completed augmentation for class: {class_name}")