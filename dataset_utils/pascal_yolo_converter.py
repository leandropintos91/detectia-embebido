import os
import xml.etree.ElementTree as ET

def convert_pascal_to_yolo(pascal_path, yolo_path, classes):
    for xml_file in os.listdir(pascal_path):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(pascal_path, xml_file))
            root = tree.getroot()

            image_width = int(root.find("size/width").text)
            image_height = int(root.find("size/height").text)

            yolo_file = os.path.splitext(xml_file)[0] + ".txt"
            with open(os.path.join(yolo_path, yolo_file), "w") as f:
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    if class_name not in classes:
                        continue

                    class_id = classes.index(class_name)

                    bbox = obj.find("bndbox")
                    x_min = int(bbox.find("xmin").text)
                    y_min = int(bbox.find("ymin").text)
                    x_max = int(bbox.find("xmax").text)
                    y_max = int(bbox.find("ymax").text)

                    x_center = (x_min + x_max) / 2.0 / image_width
                    y_center = (y_min + y_max) / 2.0 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    pascal_directory = "/mnt/d/Desktop/images"
    yolo_directory = "/mnt/d/Desktop/images/yolo"
    class_names = ["bache", "fisura", "tapa_metal", "tapa_cemento", "maxima"]  # Update with your class names

    convert_pascal_to_yolo(pascal_directory, yolo_directory, class_names)
