#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_object_perimeter_filled.py

Detección en tiempo real del contorno y cálculo de perímetro
del objeto más grande del encuadre, usando Canny + findContours,
y relleno semitransparente del área detectada.

Requisitos:
 - OpenCV (pip install opencv-python)
 - numpy

Uso:
    python3 simple_object_perimeter_filled.py
    Presiona ESC para salir.
"""

import cv2
import numpy as np

def open_camera(indices=(0,1)):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("No se pudo abrir la cámara.")

def main():
    cap = open_camera()
    window_name = "Perímetro y Relleno en Tiempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-procesado
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 1.5)

        # Detección de bordes
        edges = cv2.Canny(blur, 50, 150)

        # Cerramos huecos pequeños
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Encontrar contornos
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Elegir el contorno de mayor área
            c = max(contours, key=cv2.contourArea)

            # Crear overlay para relleno semitransparente
            overlay = frame.copy()
            cv2.drawContours(overlay, [c], -1, (0, 255, 0), thickness=cv2.FILLED)

            # Mezclar overlay con el frame original (alpha = 0.3)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Dibujar el contorno (línea más gruesa para que destaque)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            # Calcular perímetro
            perim = cv2.arcLength(c, True)
            cv2.putText(frame,
                        f"Perimetro: {perim:.1f} px",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

        # Mostrar resultado
        cv2.imshow(window_name, frame)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
