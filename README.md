#  Real-Time American Sign Language (ASL) Recognition System  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-ff4b4b)
![License](https://img.shields.io/badge/License-Academic-blueviolet)

### AI & DSML Project – Pokhara University  

A deep learning–based real-time American Sign Language (ASL) recognition system developed as part of the **Artificial Intelligence (AI)** and **Data Science & Machine Learning (DSML)** coursework at **Pokhara University**.

This project integrates computer vision, deep learning, and API-based deployment to recognize ASL alphabet gestures in real time using a webcam.

---

##  Project Overview

The system is designed to reduce communication barriers by automatically detecting and classifying ASL alphabet gestures.

###  Core Components

- **YOLOv8** – Hand detection (localization)
- **Convolutional Neural Network (CNN)** – Gesture classification
- **FastAPI** – Backend inference API
- **Streamlit** – Web-based frontend interface

---
##  System Architecture
```
┌─────────────────────────────┐
│         Webcam Input        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      YOLOv8 Hand Detection  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Hand Region Cropping  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     CNN Gesture Classifier  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     FastAPI Inference API   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Streamlit Frontend    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Predicted ASL Letter   │
└─────────────────────────────┘
```

---

## 🧠 Model Development

### 1️⃣ CNN – Gesture Classification

- Input Shape: 64 × 64 × 3  
- Output Classes: 29  
  - A–Z  
  - del  
  - space  
  - nothing  
- Activation Functions:
  - ReLU (Hidden Layers)
  - Softmax (Output Layer)
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  
- Validation Strategy: Stratified K-Fold Cross Validation  
- Data Normalization: Pixel scaling (0–1)

---

### 2️⃣ YOLOv8 – Hand Detection

- Custom-trained YOLOv8 model  
- Real-time hand localization  
- Cropped bounding box passed to CNN  
- Adjustable confidence threshold  

---

##  Dataset Description

- Dataset: ASL Alphabet Dataset  
- Total Classes: 29  
- Image Type: RGB  
- Preprocessing:
  - Image resizing to 64×64
  - Normalization
  - Data augmentation (if applied)

---

## 📈 Model Performance

| Metric      | Value (Replace with Actual Results) |
|------------|--------------------------------------|
| Accuracy   | 95%                                  |
| Precision  | 94%                                  |
| Recall     | 93%                                  |
| F1-Score   | 94%                                  |
| ROC-AUC    | 0.97                                 |

Evaluation tools used:
- Confusion Matrix
- ROC Curve
- Accuracy/Loss Graphs

---

## ⚙️ Installation
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/asl-recognition.git
cd asl-recognition

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```
