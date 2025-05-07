import cv2
import numpy as np
import glob

# === CONFIGURATION ===
image_folder = './data/*.jpg'
output_txt = 'gopro_hero11_calibration.txt'
output_npz = 'gopro_hero11_calibration.npz'

# Chessboard pattern (inner corners)
pattern_size = (7, 7)
square_size = 24  # in mm

# Prepare object points
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []
inner_corner_edge_points = {}  # To store edge points per image

images = glob.glob(image_folder)

def extract_inner_edge_points(corners2, pattern_size):
    """
    Extracts the points along the inner edges (top, bottom, left, right rows/columns).
    """
    return {
        'top_edge': corners2[0:pattern_size[0]].reshape(-1, 2),
        'bottom_edge': corners2[-pattern_size[0]:].reshape(-1, 2),
        'left_edge': corners2[::pattern_size[0]].reshape(-1, 2),
        'right_edge': corners2[pattern_size[0]-1::pattern_size[0]].reshape(-1, 2)
    }

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read image {fname}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to smooth the image while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Invert if needed (depending on which color is predominant)
    # Check which is more common: black or white pixels
    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = cv2.bitwise_not(thresh)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Try to find the chessboard corners (standard approach)
    pattern_size = (7, 7)  # For 8x8 chessboard
    ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)

    if not ret:
        # Try different parameters for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Increase contrast for better corner detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Try with enhanced image
        ret, corners = cv2.findChessboardCorners(enhanced, pattern_size, 
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                      cv2.CALIB_CB_FAST_CHECK)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"Chessboard detected in {fname}")

        edge_points = extract_inner_edge_points(corners2, pattern_size)
        inner_corner_edge_points[fname] = edge_points
    else:
        print(f"Chessboard NOT detected in {fname}")

if not objpoints:
    print("No valid calibration images found.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === Save to TXT ===
with open(output_txt, 'w') as f:
    f.write('[Intrinsic]\n')
    f.write(f'fx = {mtx[0,0]}\n')
    f.write(f'fy = {mtx[1,1]}\n')
    f.write(f'cx = {mtx[0,2]}\n')
    f.write(f'cy = {mtx[1,2]}\n')
    f.write('\n[Distortion]\n')
    f.write(f'k1 = {dist[0][0]}\n')
    f.write(f'k2 = {dist[0][1]}\n')
    f.write(f'p1 = {dist[0][2]}\n')
    f.write(f'p2 = {dist[0][3]}\n')
    f.write(f'k3 = {dist[0][4]}\n')
    f.write('\n[Image]\n')
    f.write(f'width = {gray.shape[1]}\n')
    f.write(f'height = {gray.shape[0]}\n')
    f.write('\n[Stereo]\n')
    f.write(f'baseline_mm = 184\n')

# === Save to NPZ ===
np.savez(output_npz, intrinsic_matrix=mtx, distortion_coefficients=dist, image_size=gray.shape[::-1], baseline_mm=184, inner_edge_points=inner_corner_edge_points)

print(f"Calibration complete. Results saved to {output_txt} and {output_npz}.")
