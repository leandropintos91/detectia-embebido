from PIL import Image
import os

input_folder = "/mnt/c/Users/leopi/Desktop/dataset 3"
output_folder = "/mnt/c/Users/leopi/Desktop/dataset 3"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all the PNG files in the input folder
png_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

for file in png_files:
    input_path = os.path.join(input_folder, file)
    output_file = file.replace(".png", "_mirror.png")
    output_path = os.path.join(output_folder, output_file)

    # Open the original image
    original_image = Image.open(input_path)

    # Create a mirrored version of the image
    mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Save the mirrored image to the output folder
    mirrored_image.save(output_path)

    print(f"Mirrored image saved: {output_path}")

print("Mirroring complete.")
