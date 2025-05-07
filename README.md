# 📷 GoPro Camera Calibration & Chessboard Detection Toolkit

This project contains two Python scripts designed to help you:  
✅ Calibrate a GoPro camera (or similar) using chessboard images  
✅ Detect and visualize a chessboard grid in tricky or dark photos

---

## 📁 What’s Inside?

The project includes:
- **`gopro_calibration.py`** → runs camera calibration using a folder of chessboard photos
- **`advance_chess_detection.py`** → detects a chessboard grid in a single image, even in difficult lighting or colors

---

## 💻 What Do You Need?

✅ A computer with Python installed (ask your tech team if unsure)  
✅ A folder with chessboard pattern images (e.g., pictures you took with your GoPro)  
✅ Basic Python packages installed: `opencv-python`, `numpy`, `matplotlib`

To install the necessary packages, run:

```
pip install opencv-python numpy matplotlib
```

---

## 🔧 How to Use

### 1️⃣ Calibrate Your GoPro Camera

This script processes all `.jpg` images in the `./data/` folder and generates calibration files.

Steps:
1. Place all your chessboard photos (taken with the GoPro) in a folder named `data`.
2. Run:
   ```
   python gopro_calibration.py
   ```
3. The script will detect the chessboard in each image, calculate the camera calibration, and save two result files:
   - `gopro_hero11_calibration.txt` → easy-to-read calibration numbers
   - `gopro_hero11_calibration.npz` → technical calibration data for other scripts

---

### 2️⃣ Detect a Chessboard Grid in One Photo

This script works on **one image** at a time.

Steps:
1. Put the target image (e.g., `left20.jpg`) into the `./data/` folder.
2. Update the file name in the script if needed (currently it uses `left20.jpg` by default).
3. Run:
   ```
   python advance_chess_detection.py
   ```
4. The script will:
   - Show you the step-by-step image processing.
   - Save a result image (`detected_chessboard.jpg`) showing the detected grid or corners.

---

## 💡 Tips

- Make sure the chessboard is fully visible and well-lit in your photos.
- If detection fails, try improving contrast or reducing reflections.
- These scripts are meant for **single use cases**, not for large-scale automated setups.

---

## 🛠 Need Help?

If you’re unsure how to run Python or set up the environment, reach out to a technical contact or developer. They can help install Python, set up packages, and run these scripts.

Feel free to ask for a **step-by-step guide with screenshots** or a **one-click batch script** if you want it even easier! 🚀
