import time
import sys
import numpy as np
import cv2

# Ruta a la API de CoppeliaSim
sys.path.insert(0, 'C:\\Program Files\\CoppeliaRobotics\\CoppeliaSimEdu\\programming\\zmqRemoteApi\\clients\\python\\src')

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Conexión con CoppeliaSim
client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')

sphereHandle = sim.getObject('/Sphere')
visionSensorHandle = sim.getObject('/visionSensor') 

x_position = 0.0
step = 0.01

# Bucle infinito
while True:
    # Mover la esfera
    sim.setObjectPosition(sphereHandle, -1, [x_position, 0, 0.2])
    x_position += step

    if x_position > 1.0:
        x_position = -1.0

    # Capturar imagen
    # Actualizar imagen del sensor
    sim.handleVisionSensor(visionSensorHandle)
    img, resolution = sim.getVisionSensorImg(visionSensorHandle)

    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow('Imagen del Sensor de Visión', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.05)

# Limpiar
cv2.destroyAllWindows()
