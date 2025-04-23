import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

# Define the find_ellipses function (if not imported from elsewhere)
def find_ellipses(frame, lower_hsv, upper_hsv):
    # Convert frame to HSV color space.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for colors in the desired range.
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Optional: Remove noise with morphological operations.
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        ellipse = cv2.fitEllipse(cnt)
        ellipses.append(ellipse)
    return ellipses

# Use a tkinter file dialog to select the image file.

# Initialize tkinter and hide the root window.
tk.Tk().withdraw()

# Open a file dialog to select an image file.
file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

# Load the selected image.
frame = cv2.imread(file_path)
if frame is None:
    print("Error: Could not load 'last_frame.png'")
    exit(1)

h, w, _ = frame.shape

# Define HSV thresholds for green.
lower_green = (45, 100, 100)
upper_green = (75, 255, 255)

# Call find_ellipses to get a list of ellipses.
ellipses = find_ellipses(frame, lower_green, upper_green)
if not ellipses:
    print("No ellipses detected.")
    exit(1)

# For the first ellipse in the list, get the major axis.
# ellipse[1] holds the two axis lengths.
_, axes, _ = ellipses[0]
major_axis = max(axes)

# Normalize the major axis length by the image width (giving ref_size).
normalized_major_axis = major_axis / float(w)
print("Normalized major axis:", normalized_major_axis)