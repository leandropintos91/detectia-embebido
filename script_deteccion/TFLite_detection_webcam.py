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
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from datetime import datetime
import json
import queue
import threading
import pytz
import base64
import requests
import subprocess
import uuid

from dotenv import load_dotenv

# Define and parse input arguments
parser = argparse.ArgumentParser()
home_path = os.path.expanduser("~")
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default=home_path + "detectia-embebido/script_deteccion/tf_model")
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




last_gps_data = { "latitude": 0.0, "longitude": 0.0, "speed": 0.0 }

lock = threading.Lock()

def checkSaveThreshold(arr, X):
    for num in arr:
        if num >= X:
            return True
    return False

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=1):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

def obtener_timestamp_iso8601():
    # Obtener el timestamp actual en UTC
    now_utc = datetime.now(pytz.utc)
    return now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
 

def detect_thread_function(cola_registros):
    global GRAPH_NAME
    global width
    global height
    global imH
    global imW
    global FRAME_STEP
    global bypass_speed
    global last_gps_data

    current_uuid = uuid.uuid4()

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    id_foto = 1

    # Initialize video stream
    if (VIDEO_NAME != ""):
        video = cv2.VideoCapture(VIDEO_NAME)
        imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        videostream = VideoStream(resolution=(imW,imH),framerate=1).start()
        time.sleep(1)

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    frameNumber = 0
    while True:

        if (VIDEO_NAME != "" and video.isOpened()):
            ret, frame1 = video.read()
            if not ret:
                print('Reached the end of the video!')
                break

            frameModule = frameNumber%FRAME_STEP
            frameNumber = frameNumber + 1
            if frameModule != 0:
                continue
        else:
            # Grab frame from video stream
            frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        last_known_speed=0.0
        with lock:
            last_known_speed = last_gps_data["speed"]
        if bypass_speed != True and last_known_speed <= 1:
            print("DET  - vehiculo detenido, skipping")
            continue 

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        detecciones = []


        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                print("DET  - se detecto algo, procesando...")

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                if(ymin < 80):
                    print("DET  - Out of bounds. Skipping")
                    continue
                
                if MODO_VISUAL:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (object_name != 'bache' and object_name != 'fisura'):
                    print("DET  - ni bache ni fisura. skipping")
                    continue
                    
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                if (MODO_VISUAL):
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                deteccion = {
                    "puntaje": int(scores[i]*100),
                    "clase": object_name,
                    "recuadro": {
                        "top": ymin, "right": xmax, "bottom": ymax, "left": xmin
                    }
                }
                detecciones.append(deteccion)
                print("DET  - deteccion exitosa. " + str(label))
        
        if (len(detecciones) > 0):
            timestamp = obtener_timestamp_iso8601()
            file_timestamp = timestamp.replace(".", ",").replace(":", "_") + "_" + str(id_foto)
            id_foto += 1
            path = home_path + "/detecciones/pictures/deteccion_" + file_timestamp + ".jpg"
            cv2.imwrite(path, frame)

            with lock:
                lat = last_gps_data['latitude']
                lon = last_gps_data['longitude']

            registro = {
                "file_timestamp": file_timestamp,
                "detecciones": detecciones,
                "path_foto": path,
                "fechaDeteccion": timestamp,
                "ubicacion": {
                    "latitud": lat,
                    "longitud": lon
                },
                "loteId": str(current_uuid)
            }
            registro_json = json.dumps(registro)

            print("Appending registry to queue")
            cola_registros.put(registro_json)

        if (MODO_VISUAL == True):
            cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


def save_thread_function(cola_registros):
    global verbose
    
    while True:
        registro = cola_registros.get()
        if registro is None:
            continue
        
        print("guardando registro")
        registro_json = json.loads(registro)

        file_timestamp = registro_json["file_timestamp"] #path imagen en embebido

        filename = f"{home_path}/detecciones/json/detection_{file_timestamp}.json"
        del registro_json["file_timestamp"]
        with open(filename, "w") as file:
            json.dump(registro_json, file)

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
            break

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
        except requests.exceptions.RequestException as e:
            print("SEN  - Error al hacer el POST:" + str(e))

        # Comprobar si la solicitud fue exitosa
        if response != None and response.status_code == 200:
            print("SEN  - procesado Ok")
            # After successful upload, you can delete the file
            os.remove(file_path)
            os.remove(registro_json["path_foto"])
            
        else:
            print("SEN  - ERROR code: " + str(response.status_code))
            print("SEN  - ERROR details: " + str(response.text))
            del registro_json["foto"]
            print(". Reencolando registro para reenviar")
            if response != None:
                print(str(response.text))

    #try:
        #response =  requests.post(BACKEND_PROCESAR_URL)
    #except requests.exceptions.RequestException as e:
        #print("SEN  - Error al hacer el POST:" + str(e))

def gps_thread_function():
    global last_gps_data

    while True:
        #first check that speed is not less than 5
        comando_gps = 'gpspipe -w -n 5 | grep TPV'
        gps_available = False
        try:
            resultado_gps = subprocess.run(comando_gps, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, timeout=1)
            gps_available = True
        except:
            print("GPS  - GPS not available")

        if gps_available == True and resultado_gps.returncode == 0:
            datos_gps = json.loads(resultado_gps.stdout)
            if datos_gps["mode"] == 2 or datos_gps["mode"] == 3:
                lat = datos_gps["lat"]
                lon = datos_gps["lon"]
                try:
                    speed = datos_gps['speed']
                except:
                    speed = 0.0
                last_gps_data = {"latitude": lat, "longitude": lon, "speed": speed }

        print("GPS  - Ultimo dato conocido: " + str(last_gps_data))


def has_usb_camera():
    # Attempt to open a video capture object
    cap = cv2.VideoCapture(0)

    # Check if the camera was opened successfully
    if cap.isOpened():
        cap.release()  # Release the camera capture object
        return True
    else:
        return False



if has_usb_camera():
    if(verbose):
        print("entrando en modo captura")
    # Crear cola para comunicación entre hilos
    cola_registros = queue.Queue()

    detectar_registros_thread = threading.Thread(target=detect_thread_function, args=(cola_registros,))
    guardar_registros_thread = threading.Thread(target=save_thread_function, args=(cola_registros,))
    gps_thread = threading.Thread(target=gps_thread_function)
    detectar_registros_thread.start()
    guardar_registros_thread.start()
    gps_thread.start()
    #se espera a que el hilo que genera los registros termine y luego se pone un valor final
    detectar_registros_thread.join()
    cola_registros.put(None)
else:
    if(verbose):
        print("entrando en modo envío")
    send_thread_function()

