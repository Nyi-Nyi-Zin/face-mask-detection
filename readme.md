# 😷 Face Mask Detection (Real-Time)

## 📌 Overview

This project is a **real-time face mask detection system** using a webcam.
It uses a deep learning model built with **TensorFlow/Keras** and **MobileNetV2** architecture to classify whether a person is wearing a mask or not.

The system captures live video, processes each frame, and displays:

* ✅ **With Mask**
* ❌ **No Mask**
* 📊 Confidence percentage

---

## 🚀 Features

* Real-time webcam detection
* Lightweight MobileNetV2-based model
* Stable predictions using averaging (deque)
* Confidence score display
* Simple and easy to run

---

## 🧠 Model Details

* Base Model: MobileNetV2
* Input Size: 224 × 224
* Output: Binary classification (Mask / No Mask)
* Activation: Sigmoid

The model weights are loaded from:

* `face_mask_classifier.h5` or
* `face_mask_classifier.keras`

---

## 📈 Training Results

The model was trained for 10 epochs using **Google Colab**.

Final performance:

* Training Accuracy: ~99.7%
* Validation Accuracy: ~99.5%
* Validation Loss: ~0.014

Example training log:

```
Epoch 10/10
accuracy: 0.9977 - loss: 0.0104
val_accuracy: 0.9954 - val_loss: 0.0141
```

These results indicate strong performance with minimal overfitting, as training and validation metrics are closely aligned.

> ⚠️ Note: Although accuracy is high, real-world performance may vary depending on lighting, camera quality, and dataset diversity.

---

## 🎯 How It Works

1. Capture frame from webcam
2. Resize to 224×224
3. Run prediction using trained model
4. Store recent predictions using deque
5. Compute average for smoother output
6. Display label and confidence on screen

---

## 📦 Requirements

```
opencv-python==4.11.0.86
numpy==1.26.4
tensorflow==2.21.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

* Press **q** to quit

---

## 📁 Project Structure

```
.
├── main.py
├── face_mask_classifier.h5
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

* Ensure your webcam is connected
* Make sure the model file exists in the project folder
* TensorFlow version should match training environment for best results

---

## 🇲🇲 Burmese Summary

ဒီ project က webcam ကို အသုံးပြုပြီး
mask တပ်ထား/မတပ်ထား ကို real-time detect လုပ်ပေးတဲ့ system ဖြစ်ပါတယ်။

Model ကို Google Colab မှာ train လုပ်ထားပြီး
TensorFlow + OpenCV ကို အသုံးပြုထားပါတယ်။

---

## 📜 License

This project is for learning and demonstration purposes.
