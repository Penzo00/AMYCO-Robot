#!/bin/bash

# Directory where CSVs will be saved
CSV_DIR="./CSV"

# Create the directory if it doesn't exist
mkdir -p "$CSV_DIR"

# Read each link from the file and download the corresponding CSV
while IFS= read -r url; do
    # Extract filename from the URL (everything after the last '/')
    filename=$(basename "$url")
    
    # Download the file and save it in the CSV directory
    curl -o "$CSV_DIR/$filename" "$url"
done < "links.txt"
