######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import json
import base64
import requests

from dotenv import load_dotenv

home_path = os.path.expanduser("~")

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default=home_path + "/detectia-embebido/script_deteccion/tf_model")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--verbose', action='store_true', help='Print log messages or not')
parser.add_argument('--bypass_speed', action='store_true', help='Take in count the speed or not')
parser.add_argument('--video', help='Name of the video file',
                    default='')
parser.add_argument('--gui', action='store_true', help='Show or not screen with boxes')
parser.add_argument('--framestep', help='Steps for skipping frames to accelerate the video. Default value does not accelerate the video',
                    default=1)


args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
verbose = bool(args.verbose)
bypass_speed = True
VIDEO_NAME = args.video
MODO_VISUAL = args.gui
FRAME_STEP = int(args.framestep)

print("MODELDIR: ")
print(MODEL_NAME)

load_dotenv()
BACKEND_CAPTURA_URL = os.getenv('BACKEND_CAPTURA_URL')
BACKEND_PROCESAR_URL = os.getenv('BACKEND_PROCESAR_URL')


home_path = os.path.expanduser("~")

def send_thread_function():
    global BACKEND_CAPTURA_URL
    global verbose
    
    files = os.listdir(home_path + "/detecciones/json")
    
    for filename in files:
        file_path = os.path.join(home_path + "/detecciones/json", filename)
        print("SEN - Procesando archivo" + filename)

        try:
            with open(file_path, "r") as file:
                #registro_json_as_string = json.load(file)
                registro_json = json.load(file)
            print(registro_json)
        except json.JSONDecodeError as e:
            print(f"SEN - Error decoding JSON data: {e}")
            continue

        path_foto = registro_json["path_foto"] #path imagen en embebido

        # Leer el contenido de la imagen
        print(path_foto)
        with open(path_foto, "rb") as archivo_imagen:
            print(archivo_imagen)
            contenido_imagen = archivo_imagen.read()

        # Codificar el contenido de la imagen en base64
        imagen_base64 = base64.b64encode(contenido_imagen).decode()

        #se agrega foto en base64
        registro_json["foto"] = imagen_base64

        #enviar datos a Backend
        headers = {'Content-Type': 'application/json'}

        # Realizar la solicitud GET
        response = None
        try:
            response =  requests.post(BACKEND_CAPTURA_URL, data=json.dumps(registro_json), headers=headers)
        except Exception as e:
            print("SEN  - Error al hacer el POST:" + str(e))

        # Comprobar si la solicitud fue exitosa
        if response != None and response.status_code == 200:
            print("SEN  - procesado Ok")
            # After successful upload, you can delete the file
            source_json_file_path = file_path
            destination_json_file_path = os.path.join(home_path + "/detecciones/enviados/json/" + filename)
            os.rename(source_json_file_path, destination_json_file_path)

            source_pictures_file_path = registro_json["path_foto"]
            destination_pictures_file_path = registro_json["path_foto"].replace("/detecciones/" , "/detecciones/enviados/")
            os.rename(source_pictures_file_path, destination_pictures_file_path)
        else:
            if(response != None and response.status_code != None):
                print("SEN  - ERROR code: " + str(response.status_code))
            if(response != None and response.text != None):
                print("SEN  - ERROR details: " + str(response.text))


send_thread_function()