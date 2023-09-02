#Verificar que este conectado correctamente:
-------------------------------------------

>lsusb
#ver que haya algo conectado con el chip CP2102 (como interfaz USB a UART - UART Bridge -).

#Si se quiere ver el nombre serial que se le dio para comunicarse con el modulo a traves de la UART:
>cd /dev
>ls
#Se puede ver ttyUSB0 (serie por USB)

#Instalar paquetes genericos para GPS:
-------------------------------------

[(>sudo apt-get install gpsd gpsd-clients python-gps) -> ya es viejo]

>sudo apt-get install gpsd gpsd-clients gpsd-tools
#Es el demonio que es capaz de monitorear dispositivos GPS conectados, y permite que aplicaciones puedan acceder a los datos del modulo
#y muestra los datos en un formato mas facil de analizar (el clasico es NMA0183 que utiliza la mayoria, y no es facil de entender)

#Varios comandos para su uso:
----------------------------

#arrancar servicio:
>sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock

#ver si funciona servicio:
>sudo service gpsd status

#Si no funciona, reiniciar servicio:
>sudo service gpsd restart

#Parar el servicio
>sudo killall gpsd

#Probar (monitoreo):
>cgps -s

#Aplicacion grafica:
>xgps

#Aplicacion para guardar datos:
>gpspipe

----------------------------------------------------------
Puede que el GPS no este tomando correctamente el puerto, 
verificar/agregar en el archivo gpsd para el parametro DEVICES:

>sudo nano /etc/default/gpsd

...
DEVICES=/dev/ttyUSB0
...