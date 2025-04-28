#!/usr/bin/env python3
# filepath: c:\Users\Sammy\Desktop\Python\VisionControl\resources\grid.py

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import argparse

def draw_grid_lines(img, n, color=(255, 255, 255), thickness=2):
    """
    Draws n-1 evenly spaced vertical and horizontal lines on the given image.
    
    Parameters:
        img: The input image (numpy array).
        n: Number of grid sections per axis (results in n-1 lines).
        color: Color tuple (B, G, R) for the grid lines.
        thickness: Thickness of the grid lines.
    
    Returns:
        The image with grid lines drawn.
    """
    h, w = img.shape[:2]
    dx = w / n
    dy = h / n
    
    # Draw vertical lines
    for i in range(1, n):
        x = int(round(i * dx))
        cv2.line(img, (x, 0), (x, h), color, thickness)
        
    # Draw horizontal lines.
    for i in range(1, n):
        y = int(round(i * dy))
        cv2.line(img, (0, y), (w, y), color, thickness)
        
    return img

def main():
    # Hardcoded or command-line parameters
    parser = argparse.ArgumentParser(
        description='Draw n-1 evenly spaced grid lines on an existing image.')
    parser.add_argument('--n', type=int, default=11,
                        help='Total squares per axis (draws n-1 grid lines).')
    parser.add_argument('--output', type=str, default='grid_output.png',
                        help='Output image filename.')
    parser.add_argument('--thickness', type=int, default=2,
                        help='Grid line thickness.')
    args = parser.parse_args()
    
    # Use tkinter filedialog to select an image file.
    root = tk.Tk()
    root.withdraw()  # Hide the main window.
    image_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    
    if not image_path:
        print("No image selected, exiting.")
        return
    
    # Load the image.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Draw grid lines on a copy of the image.
    img_with_grid = draw_grid_lines(img.copy(), args.n, color=(255, 255, 255), thickness=args.thickness)
    
    # Display and save the output.
    cv2.imshow("Grid", img_with_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(args.output, img_with_grid)
    print(f"Output saved as {args.output}")

if __name__ == '__main__':
    main()
