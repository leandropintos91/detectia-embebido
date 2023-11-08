#!/bin/bash
source /home/$USER/detectia-embebido/detectia-env/bin/activate
#sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock
python3 /home/$USER/detectia-embebido/script_deteccion/TFLite_menu_detection_webcam.py --verbose
