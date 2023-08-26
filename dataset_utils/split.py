import os
import random
import shutil

def split_dataset(input_dir, output_train_dir, output_val_dir, split_ratio):
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    file_list = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    random.shuffle(file_list)

    split_idx = int(len(file_list) * split_ratio)
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:]

    for file in train_files:
        base_name = os.path.splitext(file)[0]
        txt_file = base_name + ".txt"

        shutil.move(os.path.join(input_dir, file), os.path.join(output_train_dir, file))
        shutil.move(os.path.join(input_dir, txt_file), os.path.join(output_train_dir, txt_file))

    for file in val_files:
        base_name = os.path.splitext(file)[0]
        txt_file = base_name + ".txt"

        shutil.move(os.path.join(input_dir, file), os.path.join(output_val_dir, file))
        shutil.move(os.path.join(input_dir, txt_file), os.path.join(output_val_dir, txt_file))

if __name__ == "__main__":
    input_directory = "/mnt/d/Desktop/images/dataset_yolo"
    output_train_directory = "/mnt/d/Desktop/images/dataset_yolo/train"
    output_val_directory = "/mnt/d/Desktop/images/dataset_yolo/val"
    train_val_split_ratio = 0.9

    split_dataset(input_directory, output_train_directory, output_val_directory, train_val_split_ratio)
