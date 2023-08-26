# UTILS

Esta carpeta contiene distintos scripts de Python y bash para trabajar con los datasets

## mirror_augmentation.py:
Realiza copias de las imágenes que se encuentren en una carpeta, pero haciendo un volteo sobre el eje latitudinal, es decir, como si fuera reflejada en un espejo.

## pascal_yolo_converter.py:
Convierte archivos .xml del formato PascalVOC a .txt de Yolo. También trabaja con todos los archivos que se encuentren en una carpeta

## split.py:
Divide un dataset en datos de test y de training con una cierta probabilidad que se le indique.

## separate.py
Separa en un dataset de Yolo los archivos de images y de labels

## corregir.sh
Dado que uno de los programas que uso para tomar capturas de los videos agrega tildes a los nombres de los archivos, este scrip los corrige. Tensorflow no acepta caracteres alfanuméricos con tilde.

## integrity.sh
El script que tenemos de tensorflow divide automáticamente el dataset en training, validation y test. Para chequear que luego del etiquetado cada imagen .png o .jpg tenga su .xml asociado, es corre este script.
