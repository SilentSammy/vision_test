import cv2
import numpy as np
import glob

# --- Calibration settings ---
# For a typical 8x8 chessboard, there are 7 inner corners per row and column.
pattern_size = (7, 7)
# Overall chessboard size (in meters)
board_size = 0.1
# Effective square size computed from the board size.
square_size = board_size / pattern_size[0]  # approx 0.0143m per square

# Termination criteria for corner sub-pixel refinement.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, e.g. (0,0,0), (square_size,0,0), etc.
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points from all images.
object_points = []  # 3D points in real world space
image_points = []   # 2D points in image plane.

# Get list of calibration images
images = glob.glob("calibration_captures/*.png")
if not images:
    print("No calibration images found in the folder 'calibration_captures'.")
    exit(1)

print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners.
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    # If found, add object points, refine the corner locations, and add image points.
    if ret:
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners_subpix)
        object_points.append(objp)
        # Draw and display the corners for visual feedback.
        cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"Chessboard not found in {fname}")

cv2.destroyAllWindows()

# Calibrate the camera.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

print("Calibration complete.")
print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)

# Save the calibration parameters to a file.
np.savez("calibration_parameters.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("Calibration parameters saved to 'calibration_parameters.npz'.")