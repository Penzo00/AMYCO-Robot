# AMYCO: Neural Networks for Mushroom Detection and Classification

AMYCO is a project dedicated to building neural network-based tools for detecting and classifying mushrooms. The project includes a suite of Python scripts and shell utilities to preprocess data, train models, and deploy the system on a Raspberry Pi.

---

## Table of Contents

1. [Overview](#overview)
2. [File Descriptions](#file-descriptions)
    - [converter.py](#converterpy)
    - [cropper.py](#cropperpy)
    - [csv_downloader.sh](#csv_downloadersh)
    - [environment.yaml](#environmentyaml)
    - [fastervit_training.py](#fastervit_trainingpy)
    - [get_images.py](#get_imagespy)
    - [links.txt](#linkstxt)
    - [mixup.py](#mixuppy)
    - [remove_duplicates.py](#remove_duplicatespy)
    - [run.py](#runpy)
    - [size.py](#sizepy)
    - [splitter.py](#splitterpy)
    - [table.py](#tablepy)
    - [update_label_file.py](#update_label_filepy)
3. [Installation and Setup](#installation-and-setup)
4. [Steps to Set Up Automatic Environment on Raspberry Pi](#steps-to-set-up-automatic-environment-on-raspberry-pi)
5. [Contributors](#contributors)
6. [License](#license)

---

## Overview

AMYCO leverages YOLO and FasterViT architectures to process images of mushrooms for detection and classification tasks. The pipeline includes tools for downloading datasets, preprocessing images, augmenting data, and training models. This README provides a detailed description of each file and instructions for setting up and running the system.

---

## File Descriptions

### converter.py
Converts segmentation data to bounding box format for use in detection models.

### cropper.py
Uses a YOLO model to crop images, preparing them for training with FasterViT.

### csv_downloader.sh
A shell script to download CSV files from the Mushroom Observer website using links stored in `links.txt`.

### environment.yaml
Defines the `amyco` Conda environment, listing all dependencies required to run the project.

### fastervit_training.py
Trains the FasterViT model using the prepared dataset.

### get_images.py
Downloads images from the Mushroom Observer site. It filters out deprecated species and keeps the shortest name among synonyms.

### links.txt
Contains links to CSV files used by `csv_downloader.sh`.

### mixup.py
Performs mixup augmentation within a species, increasing the number of images by 10%.

### remove_duplicates.py
Checks the uniqueness of images by comparing MD5 hashes, removing duplicates to ensure data integrity.

### run.py
The main script for running the AMYCO pipeline on a Raspberry Pi 400.

### size.py
Calculates the median dimensions of images in the dataset for better model configuration.

### splitter.py
Splits the dataset into training (80%) and validation (20%) folders.

### table.py
Generates a table comparing the accuracies of various works, including this project’s results.

### update_label_file.py
Converts all YOLO class labels in the dataset to a single class (mushroom).

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/amyco.git
   cd amyco
   ```

2. Install Miniconda (if not already installed):
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-aarch64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda3
   rm ~/miniconda.sh
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   conda init
   ```

3. Create and activate the `amyco` environment:
   ```bash
   conda env create -f environment.yaml
   conda activate amyco
   ```

4. Add automatic environment activation:
   ```bash
   echo 'conda activate amyco' >> ~/.bashrc
   source ~/.bashrc
   ```

---

## Steps to Set Up Automatic Environment on Raspberry Pi

1. Download and install Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-aarch64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda3
   rm ~/miniconda.sh
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   conda init
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate amyco
   ```

3. Enable HDMI audio output (optional):
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
   Uncomment and edit the following lines:
   ```bash
   hdmi_group=1
   hdmi_mode=1
   hdmi_drive=2
   ```

4. Reboot the Raspberry Pi to apply changes.

---

## Contributors
- Chiara D'Amato
- Edoardo Torre
- Lorenzo Vergata

---

## License
This project is licensed under the MIT License. See the LICENSE file for details. (to be honest, there is no license)


