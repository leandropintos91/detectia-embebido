import os
import shutil

def separate_images_labels(input_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    file_list = os.listdir(input_dir)

    for file in file_list:
        if file.endswith(".png"):
            shutil.move(os.path.join(input_dir, file), os.path.join(output_images_dir, file))
        elif file.endswith(".txt"):
            shutil.move(os.path.join(input_dir, file), os.path.join(output_labels_dir, file))

if __name__ == "__main__":
    input_directory = "/mnt/d/Desktop/images/dataset_yolo/val"
    output_images_directory = "/mnt/d/Desktop/images/dataset_yolo/val/images"
    output_labels_directory = "/mnt/d/Desktop/images/dataset_yolo/val/labels"

    separate_images_labels(input_directory, output_images_directory, output_labels_directory)
