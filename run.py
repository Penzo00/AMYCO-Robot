import cv2
import onnxruntime as ort
from ultralytics import YOLO
import time
import numpy as np
import pandas as pd
import re
import sounddevice as sd
from piper.voice import PiperVoice

# Initialize OpenCV optimizations
cv2.setUseOptimized(True)

# Initialize audio streams
def init_stream(voice):
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()
    return stream

def play_audio(voice, text, stream):
    for audio_bytes in voice.synthesize_stream_raw(text):
        stream.write(np.frombuffer(audio_bytes, dtype=np.int16))

# Load voices
roman_voice = PiperVoice.load("it_IT-paola-medium.onnx")
english_voice = PiperVoice.load("en_GB-cori-high.onnx")

# Initialize audio streams
english_stream = init_stream(english_voice)
roman_stream = init_stream(roman_voice)

play_audio(english_voice, "Initialization started", english_stream)
# Load models
model = YOLO("best_ncnn_model", task="detect")
ort_session = ort.InferenceSession("amyco.onnx")

# Load data
class_names = np.load('classes.npy', allow_pickle=True)
names_df = pd.read_csv("CSV/names.csv", sep="\t").drop(['author', 'rank'], axis=1).fillna(-1)
names_df[['id', 'deprecated', 'correct_spelling_id', 'synonym_id']] = names_df[
    ['id', 'deprecated', 'correct_spelling_id', 'synonym_id']].astype('int32')
descriptions = pd.read_csv('CSV/name_descriptions.csv', delimiter='\t', on_bad_lines='skip').fillna("")

# Camera initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

# Detection thresholds
BEST_THRESHOLD = 0.5713947415351868
CONF_THRESHOLD = 0.5713947415351868
MIN_THRESHOLD = 0.024
play_audio(english_voice, "Initialization ended", english_stream)

# Extract bounding box with highest confidence score
def get_highest_confidence_box(results):
    boxes = results[0].boxes
    if not boxes:
        return None, 0
    max_idx = np.argmax(boxes.conf)
    x1, y1, x2, y2 = map(int, boxes.xyxy[max_idx])
    return results[0].orig_img[y1:y2, x1:x2], boxes.conf[max_idx]

# Capture a frame from the camera
def capture_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None

# Mix two images by averaging pixel values
def mix_images(img1, img2):
    max_side = max(max(img1.shape[:2]), max(img2.shape[:2]))
    return cv2.addWeighted(cv2.resize(img1, (max_side, max_side)), 0.5,
                           cv2.resize(img2, (max_side, max_side)), 0.5, 0)

# Clean and format text descriptions
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[_\[\]]', '', re.sub(r'http\S+|www\S+', '', text.replace('\\n', ' ')))).strip()

# Retrieve a description for a detected class
def get_description(name_id):
    rows = descriptions.query("name_id == @name_id").values
    return None if len(rows) == 0 else clean_text(" ".join(map(str, rows[:, 1:].flatten())))[3:]

# Preprocess the image for ONNX model input
def preprocess_image(image):
    return np.transpose(cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(224, 224),
                                              mean=(0, 0, 0), swapRB=False, crop=False), (0, 2, 3, 1))

# Function to get predictions from ONNX model
def get_onnx_predictions(image):
    preds = ort_session.run([ort_session.get_outputs()[0].name], {ort_session.get_inputs()[0].name: image})[0][0]
    top5_indices = preds.argsort()[-5:][::-1]
    return [(class_names[i], preds[i]) for i in top5_indices]

# Main frame processing loop
def process_frames():
    while True:
        frame1 = capture_frame(cap)
        if frame1 is None:
            play_audio(english_voice, "Camera not detected. Exiting...", english_stream)
            break

        results1 = model(frame1, conf=CONF_THRESHOLD)
        box1, prob1 = get_highest_confidence_box(results1)

        if prob1 <= MIN_THRESHOLD:
            play_audio(english_voice, "Searching for mushrooms...", english_stream)
            continue

        if prob1 < BEST_THRESHOLD:
            play_audio(english_voice, "There should be a mushroom.", english_stream)
            continue

        time.sleep(2)
        frame2 = capture_frame(cap)
        if frame2 is None:
            continue

        results2 = model(frame2, conf=CONF_THRESHOLD)
        box2, prob2 = get_highest_confidence_box(results2)

        final_box = preprocess_image(mix_images(box1, box2) if prob1 >= BEST_THRESHOLD and prob2 >= BEST_THRESHOLD
                                     else box1 if prob1 > prob2 else box2)

        cv2.imwrite("savedImage.jpg", np.squeeze(final_box) * 255)

        try:
            top5_preds = get_onnx_predictions(final_box)
        except Exception as e:
            print(f"ONNX inference error: {e}")
            continue

        for label, _ in top5_preds:
            play_audio(roman_voice, label, roman_stream)

            name_id = names_df.query("text_name == @label and deprecated == 0")['id'].values
            if not name_id.size:
                continue

            description = get_description(name_id[0])
            if description:
                play_audio(english_voice, description, english_stream)

# Run the detection process
try:
    process_frames()
except KeyboardInterrupt:
    play_audio(english_voice, "Detection interrupted by user.", english_stream)
finally:
    cap.release()
    english_stream.close()
    roman_stream.close()
    cv2.destroyAllWindows()
