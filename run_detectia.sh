#!/bin/bash
source tflite2-env/bin/activate
sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock
python ./script_deteccion/TFLite_detection_webcam.py --bypass_speed


