
# ASL Detector 👁️👋

A real-time American Sign Language (ASL) hand gesture detection system built using OpenCV, TensorFlow, and CVZone.

## 📸 Project Overview

This project enables:
- **Hand tracking** using the CVZone HandTrackingModule
- **Image preprocessing** for consistent input size
- **Gesture classification** using a trained deep learning model (`keras_model.h5`)
- **Live prediction display** of ASL letters (A, B, C)

## 🚀 How to Run

### 1. Clone the repo:
```bash
git clone git@github.com:wagiham/asl-detector.git
cd asl-detector
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure your system has:

* Python 3.8+
* OpenCV
* CVZone
* NumPy
* TensorFlow

### 3. Collect Training Images

Run the image collection script to capture hand gestures:

```bash
python asl-detection.py
```

Press `s` to save an image frame.

### 4. Run the Classifier

After training your model or using an existing one:

```bash
python test.py
```

Live camera feed will display predictions in real-time.

## 🧠 Model Training

The `keras_model.h5` is trained using Teachable Machine or a custom pipeline. Ensure your training images are well-lit and centered.

## 📌 Notes

* Uses webcam index `1` by default; change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` if needed.
* `HandDetector` can be configured for different detection accuracy.
* Data collection and classification are separated for modular development.

## ✍️ Author

**Wagiha Mariam** – [wagiham](https://github.com/wagiham)

---

Feel free to improve or expand this project. PRs welcome!


