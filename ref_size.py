import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

# Use a file dialog to select the image

root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

if not file_path:
    print("No file selected.")
    exit(1)

# Load the selected image
frame = cv2.imread(file_path)
if frame is None:
    print("Error: Could not load 'last_frame.png'")
    exit(1)

h, w, _ = frame.shape

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
if hasattr(cv2.aruco, 'DetectorParameters_create'):
    parameters = cv2.aruco.DetectorParameters_create()
else:
    parameters = cv2.aruco.DetectorParameters()

# Detect the markers in the image
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is None or len(corners) == 0:
    print("No markers detected.")
    exit(1)

# Process the first detected marker
marker_corners = corners[0].reshape((4, 2))
distances = []
for i in range(4):
    j = (i + 1) % 4  # next corner (with wrap-around)
    dx = marker_corners[j][0] - marker_corners[i][0]
    dy = marker_corners[j][1] - marker_corners[i][1]
    distances.append(math.hypot(dx, dy))
apparent_size_pixels = max(distances)

# Compute normalized marker size (ref_size) with respect to frame width
ref_size = apparent_size_pixels / float(w)
print("ref_size:", ref_size)