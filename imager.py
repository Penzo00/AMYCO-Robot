import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import onnxruntime as ort
import csv
import onnxruntime as ort
from ultralytics import YOLO
import time
import numpy as np
import pandas as pd
import re
import sounddevice as sd
from piper.voice import PiperVoice

# Initialize audio streams
def init_stream(voice):
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()
    return stream

def play_audio(voice, text, stream):
    for audio_bytes in voice.synthesize_stream_raw(text):
        stream.write(np.frombuffer(audio_bytes, dtype=np.int16))

# Clean and format text descriptions
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[_\[\]]', '', re.sub(r'http\S+|www\S+', '', text.replace('\\n', ' ')))).strip()

# Retrieve a description for a detected class
def get_description(name_id):
    rows = descriptions.query("name_id == @name_id").values
    return None if len(rows) == 0 else clean_text(" ".join(map(str, rows[:, 1:].flatten())))[3:]

# Load models
ort_session = ort.InferenceSession("amyco.onnx")

# Load voices
roman_voice = PiperVoice.load("it_IT-paola-medium.onnx")
english_voice = PiperVoice.load("en_GB-cori-high.onnx")

# Initialize audio streams
english_stream = init_stream(english_voice)
roman_stream = init_stream(roman_voice)

# Load data
class_names = np.load('classes.npy', allow_pickle=True)
names_df = pd.read_csv("CSV/names.csv", sep="\t").drop(['author', 'rank'], axis=1).fillna(-1)
names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']] = names_df[
    ['id', 'deprecated', 'correct_spelling_id', 'synonym_id']].astype('int32')
descriptions = pd.read_csv('CSV/name_descriptions.csv', delimiter='\t', on_bad_lines='skip').fillna("")

# Load class dictionary
class_dict = np.load("classes.npy", allow_pickle=True)

# Initialize YOLO model
yolo_model = YOLO("best_ncnn_model", task="detect")
yolo_conf = 0.024

# Initialize ONNX model
onnx_model = ort.InferenceSession("amyco.onnx")

# Function to preprocess image for ONNX model
def preprocess_onnx(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0  # Normalizza tra 0 e 1
    #image = np.transpose(image, (2, 0, 1))  # Cambia l'ordine dei canali a (C, H, W)
    image = np.expand_dims(image, axis=0)  # Aggiungi la dimensione batch
    return image

# Function to get predictions from ONNX model
def get_onnx_predictions(image):
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    preds = onnx_model.run([output_name], {input_name: image})[0]
    top5_indices = preds[0].argsort()[-5:][::-1]
    top5_preds = [(class_dict[i], preds[0][i]) for i in top5_indices]
    return top5_preds

# Directory containing images
image_dir = "Test_classifiers"
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Lists to store original and cropped images
original_images = []
cropped_images = []

# CSV file to save predictions
csv_file = open("predictions.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["File Name", "ONNX Top1", "ONNX Prob1", "ONNX Top2", "ONNX Prob2", "ONNX Top3", "ONNX Prob3", "ONNX Top4", "ONNX Prob4", "ONNX Top5", "ONNX Prob5"])

# Process each image
for image_file in image_files:
    print(image_file)
    # Read and store original image
    image_path = os.path.join(image_dir, image_file)
    original_image = Image.open(image_path).convert("RGB")
    original_images.append(original_image)

    # Perform object detection with YOLO
    results = yolo_model.predict(original_image, conf=yolo_conf)

    # Crop detected objects and store cropped images
    if len(results[0].boxes) > 0:
        cropped_image = original_image.crop(results[0][results[0].boxes.conf == max(results[0].boxes.conf)].boxes.xyxy[0].tolist())
        #cropped_image = original_images[idx][y1:y2, x1:x2]
        # Preprocess cropped image for ONNX model
        cropped_images.append(cropped_image)
        # Preprocess cropped image for ONNX model
        onnx_input = preprocess_onnx(cropped_image)

        # Get predictions from ONNX model
        onnx_preds = get_onnx_predictions(onnx_input)
        
        # Write predictions to CSV file
        csv_writer.writerow([image_file] + [item for sublist in onnx_preds for item in sublist])

csv_file.close()


# Plot original images in a 3x3 grid
plt.figure(figsize=(12, 12))
for i in range(min(9, len(original_images))):
    original_images[i] = np.asarray(original_images[i].convert("RGB"), dtype=np.uint8)
    plt.subplot(3, 3, i+1)
    plt.imshow(original_images[i])
    plt.axis('off')
plt.suptitle("Original Images", fontsize=36)
plt.savefig("original_images_plot.png")  # Save the plot
plt.show()

# Plot cropped images in a 3x3 grid
plt.figure(figsize=(12, 12))
for i in range(min(9, len(cropped_images))):  # Avoid errors with fewer than 9 images
    cropped_images[i] = np.asarray(cropped_images[i].convert("RGB"), dtype=np.uint8)
    plt.subplot(3, 3, i+1)
    plt.imshow(cropped_images[i])
    plt.axis('off')
plt.suptitle("Cropped Images", fontsize=36)
plt.savefig("cropped_images_plot.png")  # Save the plot
plt.show()
# Load predictions from CSV file for comparison plots
predictions = np.genfromtxt("predictions.csv", delimiter=',', dtype=None, encoding=None, skip_header=1)

# Plot comparison histograms for ONNX predictions
plt.figure(figsize=(18, 18))  # Set a large grid for all subplots
for i in range(min(9, len(predictions))):  # Ensure at least 9 predictions
    file_name = predictions[i][0]
    
    onnx_labels = [predictions[i][j] for j in range(1, 10, 2)]
    onnx_probs = [float(predictions[i][j]) for j in range(2, 11, 2)]
    
    x = np.arange(len(onnx_labels))
    
    plt.subplot(3, 3, i + 1)  # Use subplot for a unified image
    plt.bar(x, onnx_probs, width=0.4, label='ONNX')
    plt.xticks(x, onnx_labels, rotation=45, ha='right', fontsize=20)  # Rotated labels with larger font
    plt.xlabel('Classes', fontsize=24)
    plt.ylabel('Probabilities', fontsize=24)
    plt.title(f'{file_name}', fontsize=24)
    plt.tight_layout()  # Improve spacing

plt.suptitle("AMYCO ONNX Predictions Probabilities", y=1.02, fontsize=26)  # Add a general title with larger font
plt.savefig("onnx_predictions_plot.png", bbox_inches='tight')  # Save the plot
plt.show()