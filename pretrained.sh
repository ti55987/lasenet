#!/bin/bash

# Define the path to the zip file and the directory to extract to
zip_file_path="results/models/4prl_pretrained.zip"
extract_to_path="results/models/"

# Create the directory if it doesn't exist
mkdir -p $extract_to_path

# Unzip the file
unzip -o $zip_file_path -d $extract_to_path

echo "File unzipped successfully to $extract_to_path"