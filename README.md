# Car Damage Recognition Demo

Este repositorio contiene los códigos y modelos necesarios para levantar el servicio de "Car Damage Recognition". Los modelos están versionados mediante la herramienta DVC.

## Uso del repositorio

Solo es necesario usar la carpeta "docker" de este repositorio para levantar el servicio.

### Pasos

1. Descargar el repositorio mediante el comando
   
   ```bash
   git clone https://github.com/jhairgallardo/car-damage-recognition-demo.git
   ```

   ó usando la descarga en formato .zip

1. Entrar a la carpeta `docker/`
   
   ```bash
   cd car-damage-recognition-demo/docker
   ```

1. Construir la imagen docker
   
   ```bash
   docker build -t cardamageimage .
   ```

1. Levantar un contenedor corriendo el servicio de "Car Damage Recognition"
   
   ```bash
   docker run --name=cardamage -it -p 7011:7000 cardamageimage
   ```

2. En otro terminal, ir a la ruta raiz del repositorio y correr el ejemplo `client.py` para corroborar la respuesta del servicio
   
   ```bash
   cd car-damage-recognition-demo/
   python client.py
   ```