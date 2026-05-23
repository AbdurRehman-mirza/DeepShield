# DeepShield — AI-Based Deepfake Image Detector

A deep learning project that detects whether a face image is Real or Fake using EfficientNet-B0 and PyTorch.

---

## About the Project

DeepShield is a binary image classifier trained on 70,000 face images from the Kaggle Deepfake Detection dataset. It uses EfficientNet-B0 with transfer learning and is deployed as a simple Streamlit web app where you can upload a face image and get an instant Real/Fake verdict with confidence score.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 92.57% |
| Precision | 96.11% |
| Recall | 88.62% |
| F1-Score | 92.21% |

---

## Project Files

- `train.py` — trains the model
- `evaluate.py` — evaluates on test set and prints metrics
- `app.py` — Streamlit web application

---

## How to Run

Install dependencies:
```
pip install torch torchvision opencv-python streamlit scikit-learn numpy
```

Train the model:
```
python train.py
```

Evaluate:
```
python evaluate.py
```

Run the web app:
```
streamlit run app.py
```

---

## Dataset

Downloaded from Kaggle — Deepfake Detection Challenge dataset. Contains real and fake face images split into Train, Validation and Test folders.

---

## Tech Stack

Python, PyTorch, EfficientNet-B0, OpenCV, Streamlit, Scikit-learn, Google Colab

---

## Team

- Mirza M. AbdurRehman (24L-0845)
- Saad Ahmad (24L-0824)

Course: Artificial Intelligence — FAST-NUCES Lahore
Instructor: Sir Usman Anwar