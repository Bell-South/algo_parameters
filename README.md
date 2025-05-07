# GoPro HERO11 Black Camera Calibration and Depth Estimation

This repository contains scripts for calibrating GoPro HERO11 Black cameras and estimating depth from images using both stereo and monocular techniques. The scripts can handle datasets from single or dual camera setups, and automatically determine the best parameters for depth estimation.

## Overview

The workflow consists of:

1. **Camera Calibration**: Calculate intrinsic and extrinsic camera parameters using chessboard patterns
2. **Stereo Calibration**: For dual-camera setups, calculate the relationship between cameras
3. **Depth Estimation**: Generate depth maps from input images using either stereo or monocular methods
4. **Object Detection**: Measure distances to objects in the scene

## Requirements

Install the required dependencies:

```bash
pip install numpy opencv-python matplotlib torch torchvision
```

For monocular depth estimation, MiDaS will be downloaded automatically when first used.

## Camera Setup

For best results:

- **Stereo Setup**: Mount two GoPro HERO11 Black cameras side by side with a baseline (distance between cameras) of 10-15cm
- **Monocular Setup**: Mount a single GoPro HERO11 Black at a known height from the ground with a fixed tilt angle

## Script Descriptions

The repository includes the following scripts:

### 1. `gopro_calibration.py`

Calculates calibration parameters for GoPro cameras using chessboard patterns.

```bash
python gopro_calibration.py --left_imgs path/to/left/images --right_imgs path/to/right/images --pattern_size 7x7 --square_size 24.0 --output_dir calibration_results --visualize
```

Key arguments:
- `--left_imgs`: Path to left/single camera images
- `--right_imgs`: Path to right camera images (for stereo, optional)
- `--pattern_size`: Inner corner count of chessboard pattern (width x height)
- `--square_size`: Chessboard square size in mm
- `--baseline`: Camera baseline in mm (if known)
- `--output_dir`: Output directory for calibration files
- `--visualize`: Flag to visualize detected corners

### 2. `depth_estimation.py`

Estimates depth from input images using stereo or monocular methods.

```bash
python depth_estimation.py --left_img path/to/left/image.jpg --right_img path/to/right/image.jpg --calib_dir calibration_results --method auto --output_dir depth_results --visualize --measure_object
```

Key arguments:
- `--left_img`: Path to left/single camera image
- `--right_img`: Path to right camera image (for stereo, optional)
- `--calib_dir`: Directory with calibration files
- `--method`: Depth estimation method (`stereo`, `mono`, or `auto`)
- `--output_dir`: Output directory for depth results
- `--visualize`: Flag to visualize results
- `--measure_object`: Flag to measure object distances

### 3. `process_datasets.py`

Process multiple calibration datasets and determine the best parameters.

```bash
python process_datasets.py --dataset_dirs dataset1 dataset2 dataset3 --output_dir calibration_results --pattern_size 7x7 --square_size 24.0 --visualize
```

Key arguments:
- `--dataset_dirs`: Directories containing calibration datasets
- `--output_dir`: Output directory for calibration results
- `--pattern_size`: Inner corner count of chessboard pattern (width x height)
- `--square_size`: Chessboard square size in mm
- `--baseline`: Camera baseline in mm (if known)
- `--visualize`: Flag to visualize the calibration process

### 4. `complete_workflow.py`

Complete workflow for camera calibration and depth estimation.

```bash
python complete_workflow.py --mode all --dataset_dirs dataset1 dataset2 dataset3 --test_left_img test_left.jpg --test_right_img test_right.jpg --output_dir results --visualize
```

Key arguments:
- `--mode`: Mode to run (`calibrate`, `depth`, or `all`)
- `--dataset_dirs`: Directories containing calibration datasets
- `--test_left_img`: Path to test left/single camera image
- `--test_right_img`: Path to test right camera image (for stereo)
- `--method`: Depth estimation method (`stereo`, `mono`, or `auto`)
- `--output_dir`: Output directory for results
- `--visualize`: Flag to visualize results

## Workflow Steps

### Step 1: Prepare Calibration Datasets

1. Print a chessboard pattern (8x8 squares, 24mm per square)
2. Take 20-30 photos of the chessboard from different angles with each camera
3. Organize images in separate directories for each dataset

### Step 2: Run Camera Calibration

```bash
python complete_workflow.py --mode calibrate --dataset_dirs dataset1 dataset2 dataset3 --output_dir results --visualize
```

This will:
- Process all datasets
- Calibrate individual cameras
- Perform stereo calibration (if applicable)
- Determine the best parameters
- Save calibration files to the output directory

### Step 3: Run Depth Estimation

```bash
python complete_workflow.py --mode depth --test_left_img test_left.jpg --test_right_img test_right.jpg --output_dir results --visualize
```

This will:
- Load the best calibration parameters
- Estimate depth from test images
- Measure object distances
- Save and visualize depth maps

### Step 4: Complete Workflow

To run both calibration and depth estimation in one go:

```bash
python complete_workflow.py --mode all --dataset_dirs dataset1 dataset2 dataset3 --test_left_img test_left.jpg --test_right_img test_right.jpg --output_dir results --visualize
```

## Example Results

After running the workflow, the following files will be generated:

- **Calibration parameters**: `*_calib.pkl` and `mono_params.pkl`
- **Comparison plots**: `camera_errors_comparison.png` and `stereo_baselines_comparison.png`
- **Depth maps**: `depth_map_*.png`
- **Visualizations**: `visualization_*.png`

## Notes and Tips

1. **Image Resolution**: For consistent results, use the same resolution for calibration and test images.
2. **Chessboard Coverage**: During calibration, try to cover the entire field of view with the chessboard pattern.
3. **Stable Mounting**: Ensure cameras are firmly mounted to maintain a consistent baseline.
4. **Lighting Conditions**: Calibrate in well-lit environments for better corner detection.
5. **Stereo vs. Monocular**: Stereo depth estimation is more accurate but requires two cameras. Monocular estimation is a good fallback option.

## Troubleshooting

- **Corner Detection Issues**: Try adjusting lighting conditions or using a larger chessboard pattern.
- **MiDaS Errors**: Ensure PyTorch is installed and your system has enough memory for the model.
- **Parameter Inconsistencies**: If stereo calibration results in unreasonable baseline values, check your camera mounting and re-calibrate.

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [MiDaS Monocular Depth Estimation](https://github.com/intel-isl/MiDaS)
- [GoPro HERO11 Black Technical Specifications](https://gopro.com/en/us/shop/cameras/hero11-black/CHDHX-111-master.html)