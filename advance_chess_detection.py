import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_chessboard_adaptive(image_path, show_steps=True):
    """
    Advanced chessboard detection for challenging cases like brown/black boards
    
    Args:
        image_path: Path to the image file
        show_steps: Whether to display intermediate processing steps
    """
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not read image from {image_path}")
        return False, None
    
    # Create a copy for visualization
    img = original.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display original grayscale
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()
    
    # Apply bilateral filter to smooth the image while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Display threshold result
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("Adaptive Threshold")
        plt.axis('off')
        plt.show()
    
    # Invert if needed (depending on which color is predominant)
    # Check which is more common: black or white pixels
    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = cv2.bitwise_not(thresh)
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(thresh, cmap='gray')
            plt.title("Inverted Threshold")
            plt.axis('off')
            plt.show()
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("After Morphological Operations")
        plt.axis('off')
        plt.show()
    
    # Try to find the chessboard corners (standard approach)
    pattern_size = (7, 7)  # For 8x8 chessboard
    ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)
    
    # If standard approach fails, try with different parameters
    if not ret:
        # Try different parameters for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Increase contrast for better corner detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(enhanced, cmap='gray')
            plt.title("Contrast Enhanced")
            plt.axis('off')
            plt.show()
        
        # Try with enhanced image
        ret, corners = cv2.findChessboardCorners(enhanced, pattern_size, 
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                      cv2.CALIB_CB_FAST_CHECK)
    
    # If still not detected, try Hough transform to find lines and intersections
    if not ret:
        print("Standard detection failed. Attempting Hough line detection...")
        
        # Use Canny edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(edges, cmap='gray')
            plt.title("Canny Edges")
            plt.axis('off')
            plt.show()
        
        # Use Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            print(f"Found {len(lines)} lines")
            
            # Separate into horizontal and vertical lines
            h_lines = []
            v_lines = []
            
            for line in lines:
                rho, theta = line[0]
                # Classify as horizontal or vertical based on theta
                if abs(theta) < 0.3 or abs(theta - np.pi) < 0.3:
                    v_lines.append((rho, theta))
                elif abs(theta - np.pi/2) < 0.3:
                    h_lines.append((rho, theta))
            
            print(f"Horizontal lines: {len(h_lines)}, Vertical lines: {len(v_lines)}")
            
            # Draw lines on the image
            line_img = img.copy()
            for rho, theta in h_lines + v_lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if show_steps:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
                plt.title("Detected Lines")
                plt.axis('off')
                plt.show()
            
            # Custom grid estimation based on line intersections
            if len(h_lines) >= 7 and len(v_lines) >= 7:
                print("Sufficient lines detected to estimate a chessboard grid")
                # Sort lines by position
                h_lines.sort()
                v_lines.sort()
                
                # Save the processed image
                cv2.imwrite("detected_chessboard_grid.jpg", line_img)
                print("Grid estimation saved as detected_chessboard_grid.jpg")
                return True, line_img
            else:
                print("Not enough lines detected for a chessboard grid")
        else:
            print("No lines detected using Hough transform")
    
    # If corners were detected using the standard approach
    if ret:
        print("Chessboard corners detected!")
        
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        
        # Display the result
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Chessboard Corners")
        plt.axis('off')
        plt.show()
        
        # Save the result
        cv2.imwrite("detected_chessboard.jpg", img)
        print("Result saved as detected_chessboard.jpg")
        
        return True, img
    
    print("Could not detect the chessboard pattern")
    return False, None

def main():
    image_path = "./data/left20.jpg"  # Update with your image path
    
    print("Attempting advanced chessboard detection...")
    detected, result_img = detect_chessboard_adaptive(image_path)
    
    if not detected:
        print("\nDetection suggestions:")
        print("1. Check that the entire chessboard is visible with clear edges")
        print("2. Ensure good lighting with minimal glare")
        print("3. Take the photo straight-on to minimize perspective distortion")
        print("4. Make sure the chessboard has clearly visible contrasting squares")
    
if __name__ == "__main__":
    main()