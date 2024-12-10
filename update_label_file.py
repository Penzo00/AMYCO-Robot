import os

# Define paths to the label folders
train_label_path = "YOLO11/detection_training_final/train/labels"
val_label_path = "YOLO11/detection_training_final/valid/labels"
test_label_path = "YOLO11/detection_training_final/test/labels"

# List of label folders to iterate over
label_paths = [train_label_path, val_label_path, test_label_path]

# Function to modify the class label in each label file
def update_label_file(file_path, new_class_id="0"):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the first element (class index) in each line
    modified_lines = []
    for line in lines:
        elements = line.strip().split()
        elements[0] = new_class_id  # Change the first element to '0'
        modified_lines.append(' '.join(elements) + '\n')

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

# Iterate through all the label folders and update the class ID in each .txt file
for label_path in label_paths:
    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_path, filename)
            update_label_file(file_path)

print("All class labels changed to 0 successfully!")