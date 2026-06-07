# Helmet Compliance Detection using CNN and OpenCV

## Project Overview

This project detects whether a person is wearing a helmet or not using a Convolutional Neural Network (CNN) and OpenCV.

The model is trained on two classes:

- Helmet
- Person Without Helmet

The trained model is integrated with a webcam to perform real-time helmet detection.

---

## Features

- Image Classification using CNN
- Real-time Webcam Detection
- TensorFlow/Keras Model
- OpenCV Integration
- Binary Classification (Helmet / No Helmet)

---

## Dataset Structure

Helmet_Dataset/

├── Helmet/

└── person_no_helmet/

### Dataset Statistics

| Class | Images |
|---------|---------|
| Helmet | 435 |
| Person No Helmet | 420 |
| Total | 855 |

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-Learn

---

## CNN Architecture

Input Image Size:

224 × 224 × 3

Architecture:

Conv2D(32) + ReLU

MaxPooling2D

Conv2D(64) + ReLU

MaxPooling2D

Conv2D(128) + ReLU

MaxPooling2D

GlobalAveragePooling2D

Dense(64) + ReLU

Dense(1) + Sigmoid

---

## Training Details

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 40
- Train-Test Split: 80-20

---

## Model Performance

Training Accuracy: ~82%

Validation Accuracy: ~75%

The model successfully identifies:

- Helmet
- No Helmet

in real-time webcam streams.

---

## Project Workflow

Dataset Collection

↓

Image Preprocessing

↓

CNN Model Training

↓

Model Evaluation

↓

Model Saving

↓

Real-Time Webcam Integration

↓

Helmet Detection

---

## Run Training

```bash
python app.py
```

## Run Prediction on Image

```bash
python prediction.py
```

## Run Real-Time Webcam Detection

```bash
python camera.py
```

---

## Future Enhancements

- YOLO-based Helmet Detection
- Rider Detection
- Traffic Surveillance Integration
- Mobile Application Deployment
- Helmet Violation Monitoring System

---

## Author

Omkar Kamate

Final Year Engineering Student

Machine Learning & Data Analytics Enthusiast