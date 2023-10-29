
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
import ST7735
import RPi.GPIO as GPIO
import time
import subprocess
import os
import requests

def check_internet_connection():
    try:
        # Intenta hacer una solicitud a un servidor en línea
        response = requests.get("https://www.google.com", timeout=5)
        
        # Si la solicitud se realizó con éxito, se considera que hay conexión a Internet
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException:
        # Si hay un error en la solicitud, se considera que no hay conexión a Internet
        return False
        
def cambiarPantalla(opcion):
	
	if opcion == 1:
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(15, 200, 15))
		draw = ImageDraw.Draw(img)
		draw.text((29, 40), "Iniciar", font=font, fill=(255, 255, 255))
		draw.text((19, 60), "captura", font=font, fill=(255, 255, 255))
		disp.display(img)
	elif opcion == 2:
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(230, 15, 5))
		draw = ImageDraw.Draw(img)
		draw.text((29, 40), "Enviar", font=font, fill=(255, 255, 255))
		draw.text((31, 60), "datos", font=font, fill=(255, 255, 255))
		disp.display(img)
	elif opcion == 3:
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(30, 170, 230))
		draw = ImageDraw.Draw(img)
		draw.text((26, 40), "Apagar", font=font, fill=(255, 255, 255))
		draw.text((21, 60), "sistema", font=font, fill=(255, 255, 255))
		disp.display(img)
        
    time.sleep(0.2)


def ejecutar(opcion):
	#print (opcion)
	
	time.sleep(0.4)
	
	if opcion == 1: #Iniciar captura
		
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
		font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)

		comandoCaptura = "python3 TFLite_detection_webcam.py --verbose"  # comando de script de captura
		# Ejecuta el proceso y obtén su información, incluido el PID
		procesoCaptura = subprocess.Popen(comandoCaptura, shell=True)
		# Guarda el PID en una variable
		pidCaptura = procesoCaptura.pid
		
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(15, 200, 15))
		draw = ImageDraw.Draw(img)
		draw.text((15, 40), "Capturando...", font=font2, fill=(255, 255, 255))
		draw.text((22, 60), "(OK para", font=font2, fill=(255, 255, 255))
		draw.text((24, 80), "terminar)", font=font2, fill=(255, 255, 255))
		disp.display(img)		
		
		while GPIO.input(boton_pin_16) != GPIO.HIGH:
			time.sleep(0.25)
		
		os.system("kill -9 " + str(pidCaptura))  
		
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(15, 200, 15))
		draw = ImageDraw.Draw(img)
		draw.text((29, 40), "Iniciar", font=font, fill=(255, 255, 255))
		draw.text((19, 60), "captura", font=font, fill=(255, 255, 255))
		disp.display(img)
		
	elif opcion == 2:
		
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
		font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
		font3 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
		
		if check_internet_connection(): #hay conexion a internet, se puede enviar
			
			img = Image.new('RGB', (WIDTH, HEIGHT), color=(230, 15, 5))
			draw = ImageDraw.Draw(img)
			draw.text((16, 40), "Enviando", font=font, fill=(255, 255, 255))
			draw.text((23, 60), "datos...", font=font, fill=(255, 255, 255))
			disp.display(img)
			
			
			# Comando que deseas ejecutar en segundo plano
			comandoEnvio = "python3 TFLite_detection_webcam.py --verbose"

			# Ejecuta el comando en segundo plano
			procesoEnvio = subprocess.Popen(comandoEnvio, shell=True)

			# Espera a que el proceso termine y obtén el código de salida
			codigo_salida_envio = procesoEnvio.wait()
			
			#print ("codigo envio: " + str(codigo_salida_envio))
			
			if codigo_salida_envio == 0: #el script de envio termino bien
				img = Image.new('RGB', (WIDTH, HEIGHT), color=(230, 15, 5))
				draw = ImageDraw.Draw(img)
				draw.text((32, 40), "Envio", font=font, fill=(255, 255, 255))
				draw.text((18, 60), "Exitoso!", font=font, fill=(255, 255, 255))
				draw.text((35, 100), "(presione OK)", font=font3, fill=(255, 255, 255))
				disp.display(img)
			else:
				#hubo problema en el envio, se puede diferenciar los problemas
				#dependiendo de la respuesta (return) del script de envio
				img = Image.new('RGB', (WIDTH, HEIGHT), color=(10, 10, 210))
				draw = ImageDraw.Draw(img)
				draw.text((34, 40), "Error:", font=font, fill=(255, 255, 255))
				draw.text((28, 65), "Problema", font=font2, fill=(255, 255, 255))
				draw.text((28, 80), "en envio!", font=font2, fill=(255, 255, 255))
				draw.text((35, 100), "(presione OK)", font=font3, fill=(255, 255, 255))
				disp.display(img)
			
		else:
			
			img = Image.new('RGB', (WIDTH, HEIGHT), color=(10, 10, 210))
			draw = ImageDraw.Draw(img)
			draw.text((34, 40), "Error:", font=font, fill=(255, 255, 255))
			draw.text((35, 65), "No hay", font=font2, fill=(255, 255, 255))
			draw.text((27, 80), "internet!", font=font2, fill=(255, 255, 255))
			draw.text((35, 100), "(presione OK)", font=font3, fill=(255, 255, 255))
			disp.display(img)
			
		while GPIO.input(boton_pin_16) != GPIO.HIGH:
			time.sleep(0.25)
			
		img = Image.new('RGB', (WIDTH, HEIGHT), color=(230, 15, 5))
		draw = ImageDraw.Draw(img)
		draw.text((29, 40), "Enviar", font=font, fill=(255, 255, 255))
		draw.text((31, 60), "datos", font=font, fill=(255, 255, 255))
		disp.display(img)
		
	elif opcion == 3:#se apaga la rasp
		
		os.system("sudo shutdown -h now") 
		
	time.sleep(0.2)
		
		
disp = ST7735.ST7735(port=0, cs=0, dc=24, backlight=None, width = 125, rst=25, rotation=90, invert=False)

WIDTH = disp.width
HEIGHT = disp.height

#imagen de detectia
image = Image.open('logo_detectia.bmp')
image = image.rotate(0).resize((WIDTH, HEIGHT))
disp.display(image)
time.sleep(5)

#Imagen de inicio de sistema
img = Image.new('RGB', (WIDTH, HEIGHT), color=(150, 0, 150))
draw = ImageDraw.Draw(img)

# Load default font.
#font = ImageFont.load_default()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

# Write some text
draw.text((15, 40), "Iniciando", font=font, fill=(255, 255, 255))
draw.text((20, 60), "sistema", font=font, fill=(255, 255, 255))

# Write buffer to display hardware, must be called to make things visible on the
# display!
disp.display(img)

# Configura el modo de pines GPIO
GPIO.setmode(GPIO.BCM)

# Define el pin GPIO para el boton
boton_pin_16 = 16 #Boton OK
boton_pin_20 = 20 #Boton abajo
boton_pin_21 = 21 #Boton arriba

# Configura el pin como entrada con resistencia pull-down interna
GPIO.setup(boton_pin_16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(boton_pin_20, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(boton_pin_21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

time.sleep(2)

img = Image.new('RGB', (WIDTH, HEIGHT), color=(15, 200, 15))
draw = ImageDraw.Draw(img)
draw.text((29, 40), "Iniciar", font=font, fill=(255, 255, 255))
draw.text((19, 60), "captura", font=font, fill=(255, 255, 255))
disp.display(img)

opcion = 1 #la pantalla de iniciar captura

try:
	while True:
        # Lee el estado del botón (activo alto)
		boton_presionado_16 = GPIO.input(boton_pin_16) == GPIO.HIGH
		boton_presionado_20 = GPIO.input(boton_pin_20) == GPIO.HIGH
		boton_presionado_21 = GPIO.input(boton_pin_21) == GPIO.HIGH

		if boton_presionado_16:
			ejecutar(opcion)
		
		elif boton_presionado_20:
			if opcion == 3:
				opcion = 1
			else:
				opcion = opcion + 1
			cambiarPantalla(opcion)
		
		elif boton_presionado_21:
			if opcion == 1:
				opcion = 3
			else:
				opcion = opcion - 1
			cambiarPantalla(opcion)

		time.sleep(0.1)  # Espera para evitar la detección de rebotes

except KeyboardInterrupt:
	GPIO.cleanup()  # Limpia los pines GPIO cuando se interrumpe el programa

