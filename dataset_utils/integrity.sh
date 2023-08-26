#!/bin/bash

# Specify the folder containing the pictures and XML files
folder="."

# Iterate over each picture file in the folder
for picture_file in "$folder"/*.png; do
    # Get the picture file name without extension
    picture_name="${picture_file%.*}"

    # Construct the XML file path
    xml_file="$picture_name.xml"

    # Check if the XML file exists
    if [ ! -f "$xml_file" ]; then
        echo "Error: XML file not found for $picture_file"
        exit 1
    fi
done

echo "All picture files have corresponding XML files."