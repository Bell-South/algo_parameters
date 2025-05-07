import numpy as np
import pickle
import os
import argparse

"""
Script to generate default calibration parameters for GoPro HERO11 Black cameras.
These are estimated parameters based on camera specifications and typical values
for action cameras with wide-angle lenses.
"""

def main():
    parser = argparse.ArgumentParser(description='Generate Default GoPro HERO11 Black Parameters')
    
    # Input arguments
    parser.add_argument('--resolution', default='5312x3552', 
                       help='Camera resolution (WxH)')
    parser.add_argument('--output_dir', default='calibration_results', 
                       help='Output directory for calibration files')
    parser.add_argument('--preset', choices=['wide', 'linear', 'narrow'], default='wide',
                       help='GoPro field of view preset')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Generate default parameters
    camera_params, mono_params = generate_default_params(resolution, args.preset)
    
    # Save parameters
    camera_file = os.path.join(args.output_dir, 'default_camera_calib.pkl')
    with open(camera_file, 'wb') as f:
        pickle.dump(camera_params, f)
    print(f"Default camera parameters saved to {camera_file}")
    
    mono_file = os.path.join(args.output_dir, 'default_mono_params.pkl')
    with open(mono_file, 'wb') as f:
        pickle.dump(mono_params, f)
    print(f"Default monocular parameters saved to {mono_file}")
    
    # Print parameters
    print_camera_params(camera_params, f"GoPro HERO11 Black ({args.preset})")
    print_mono_params(mono_params)


def generate_default_params(resolution=(5312, 3552), preset='wide'):
    """
    Generate default camera parameters for GoPro HERO11 Black.
    
    Args:
        resolution: Camera resolution (width, height)
        preset: GoPro field of view preset ('wide', 'linear', 'narrow')
        
    Returns:
        Tuple of (camera_params, mono_params)
    """
    width, height = resolution
    
    # Default parameters based on typical GoPro values
    # These are approximates and should be adjusted based on your specific camera
    
    # Focal length factor based on preset
    focal_factor = {
        'wide': 0.75,    # Wide FOV (~150°)
        'linear': 1.0,   # Linear FOV (~90°)
        'narrow': 1.5    # Narrow FOV (~65°)
    }
    
    # Distortion factor based on preset
    distortion_factor = {
        'wide': 1.0,      # Strong distortion
        'linear': 0.5,    # Medium distortion
        'narrow': 0.25    # Low distortion
    }
    
    # Calculate focal length based on resolution and preset
    fx = width * focal_factor[preset]
    fy = fx  # Same focal length for both axes
    
    # Principal point at center
    cx = width / 2
    cy = height / 2
    
    # Create camera matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Distortion coefficients for wide-angle lens
    # k1, k2, p1, p2, k3
    k1 = -0.22 * distortion_factor[preset]  # Barrel distortion
    k2 = 0.05 * distortion_factor[preset]   # Higher order distortion
    p1 = 0  # Assuming no tangential distortion
    p2 = 0  # Assuming no tangential distortion
    k3 = 0  # Higher order distortion (usually small)
    
    dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
    
    # Calculate horizontal FOV
    fov_horizontal = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    
    # Camera parameters
    camera_params = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'image_size': resolution,
        'error': 0.0,  # No error because these are default parameters
        'preset': preset
    }
    
    # Monocular parameters for depth estimation
    mono_params = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'camera_height': 1.2,  # Default camera height in meters (adjust as needed)
        'tilt_angle': 15.0,    # Default tilt angle in degrees (adjust as needed)
        'fov_horizontal': fov_horizontal,
        'image_size': resolution,
        'preset': preset
    }
    
    return camera_params, mono_params


def print_camera_params(params, camera_name="GoPro HERO11 Black"):
    """
    Print camera parameters in a readable format.
    
    Args:
        params: Camera parameters dictionary
        camera_name: Name of the camera
    """
    print(f"\n{camera_name} Default Parameters:")
    print(f"  Image Size (width x height): {params['image_size'][0]} x {params['image_size'][1]} pixels")
    
    # Camera matrix
    mtx = params['camera_matrix']
    print("  Camera Matrix:")
    print(f"    fx: {mtx[0, 0]:.6f} pixels")
    print(f"    fy: {mtx[1, 1]:.6f} pixels")
    print(f"    cx: {mtx[0, 2]:.6f} pixels")
    print(f"    cy: {mtx[1, 2]:.6f} pixels")
    
    # Distortion coefficients
    dist = params['dist_coeffs']
    print("  Distortion Coefficients:")
    print(f"    k1: {dist[0, 0]:.6f}")
    print(f"    k2: {dist[0, 1]:.6f}")
    print(f"    p1: {dist[0, 2]:.6f}")
    print(f"    p2: {dist[0, 3]:.6f}")
    print(f"    k3: {dist[0, 4]:.6f}")
    
    # Calculate FOV
    fx = mtx[0, 0]
    width = params['image_size'][0]
    fov = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    print(f"  Estimated Horizontal Field of View: {fov:.2f} degrees")
    
    # Preset
    if 'preset' in params:
        print(f"  Preset: {params['preset']}")


def print_mono_params(params):
    """
    Print monocular parameters in a readable format.
    
    Args:
        params: Monocular parameters dictionary
    """
    print("\nMonocular Parameters for Depth Estimation:")
    print(f"  Camera Height: {params['camera_height']:.2f} meters (default, adjust as needed)")
    print(f"  Tilt Angle: {params['tilt_angle']:.2f} degrees (default, adjust as needed)")
    print(f"  Horizontal Field of View: {params['fov_horizontal']:.2f} degrees")
    print(f"  Preset: {params['preset']}")


def generate_stereo_params(left_params, right_params, baseline=120.0):
    """
    Generate stereo parameters from individual camera parameters.
    
    Args:
        left_params: Left camera parameters
        right_params: Right camera parameters
        baseline: Distance between cameras in mm
        
    Returns:
        Dictionary with stereo parameters
    """
    # Extract camera matrices and distortion coefficients
    left_mtx = left_params['camera_matrix']
    left_dist = left_params['dist_coeffs']
    right_mtx = right_params['camera_matrix']
    right_dist = right_params['dist_coeffs']
    image_size = left_params['image_size']
    
    # Default rotation and translation
    # Assuming cameras are perfectly aligned horizontally
    R = np.eye(3)  # Identity rotation matrix
    T = np.array([[baseline], [0], [0]])  # Translation along X-axis
    
    # Calculate essential and fundamental matrices
    E = np.cross(T.reshape(3), R.reshape(9)).reshape(3, 3)  # Approximate essential matrix
    F = np.linalg.inv(right_mtx.T) @ E @ np.linalg.inv(left_mtx)  # Approximate fundamental matrix
    
    # Calculate rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_mtx, left_dist, right_mtx, right_dist, image_size, R, T
    )
    
    # Return stereo parameters
    return {
        'left_camera_matrix': left_mtx,
        'left_dist_coeffs': left_dist,
        'right_camera_matrix': right_mtx,
        'right_dist_coeffs': right_dist,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi1': roi1,
        'roi2': roi2,
        'baseline': baseline,
        'image_size': image_size
    }


if __name__ == "__main__":
    main()