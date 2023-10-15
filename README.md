# Detectia - Sistema Embebido

Acá se detallan los pasos para instalar dependencias y requerimientos para ejecutar el script de detección y de envío de capturas de DetectIA

## Configuración previa

La distribución que debe utilizarse en la Raspberry Pi 4 es Debian 11 (bullseye). Es muy importante esto para evitar problemas de compatibilidad.
Una vez dentro del sistema, asegurarse de que el sistema esté actualizado. Para ello ejecutar: `sudo apt-get update` y `sudo apt-get upgrade`

### a) Instalar Python 3.9
La versión de Python que se usa es 3.9.
Para instalarla, ejecutar los siguientes pasos:

1. `sudo apt-get install libssl-dev openssl make gcc`
2. `cd /opt`
3. `sudo wget https://www.python.org/ftp/python/3.9.2/Python-3.9.2.tgz`
4. `sudo tar xzvf Python-3.9.2.tgz`
5. `cd Python-3.9.2`
6. `sudo ./configure`
7. `sudo make`
8. `sudo make install`
9. `sudo ln -fs /opt/Python-3.9.2/Python /usr/bin/python3.9`

Finalmente verificar la instalación con el siguiente comando: `python3.9 --version`

### b) Crear un ambiente virtual
1. Para evitar conflictos con las dependencias usamos ambientes virtuales en python. Para crear uno ejecutar: `python3.9 -m venv detectia-env`. "detectia-env" es el nombre del ambiente virtual, esto creará una carpeta con el ambiente configurado.
2. Todas las dependencias que se instalen y scripts que se corran deben estar instaladas dentro de ese ambiente virtual. Para activarlo, ejecutar: `source detectia-env/bin/activate`

### c) Crear carpeta de datos
Los datos de las capturas y detecciones se van a guardar en una estructura específica de carpetas.
Crear en el home una carpeta que se llame "detecciones" y dentro de ella otras dos que se llamen "pictures" y "json".

## Instalación de dependencias

1. Con el ambiente virtual activado, instalar las dependencias del script. Las dependencias están especificadas en el archivo `./script_deteccion/get_pi_requirements.sh`. Las dependencias incluyen bibliotecas de TensorFlow LIte, OpenCV y otras bibliotecas auxiliares para el funcionamiento del script. Para ejecutarlo, ejecutar el siguiente comando: `bash ./script_deteccion/get_pi_requirements.sh`.

Se van a descargar e instalar varios GB de dependencias. Tarda aproximadamente media hora.

## Solución a problemas comunes

### OpenCV

Si algún error ocurre al iniciar el script a causa de OpenCV relacionado a algo de multiarray, es por un problema de dependencias con numpy.

Solución: Hacer un downgrade a 1.22.3. Desinstalar numpy con `pip3 uninstall numpy` y luego instalar la versión correcta con `pip3 install numpy==1.22.3`

## Ejecución y uso del script

Para ejecutar el script usar el comando `./run_detectia`. Este comando automáticamente inicializará el GPS, y ejecutará el script `./script_deteccion/TFLite_deteccion_webcam.py`. Esto entrará en dos posibles modos: envío o detección.
- Si la webcam está conectada, automáticamente entrará en modo captura y guardará las capturas en las carpetas` ~/detecciones/pictures` y `~/detecciones/json`, las fotos y los metadatos respectivamente.
- Si la webcam está desconectada, automáticamente entrará en modo envío y enviará al backend (cuyo host está configurado en el archivo .env) todas las detecciones almacenadas, borrando las que sean exitosas y conservando las que no puedan procesarse.

