# Sign Language Recognition using DenseNet — Deep Learning Approach

## 📌 Project Overview
Real-time American Sign Language (ASL) alphabet recognition using 
DenseNet deep learning architecture. The system classifies 29 hand 
gestures (A-Z + space, delete, nothing) from images.

## 📊 Dataset
- **ASL Alphabet Dataset** — 87,000 images, 29 classes
- **Source**: [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Image Size**: 200×200 RGB

## 🏗️ Project Stages
| Week | Stage | Status |
|------|-------|--------|
| 1 | Dataset Setup & EDA | ✅ Complete |
| 2 | Data Preprocessing & Augmentation | 🔲 Pending |
| 3 | DenseNet Model Architecture | 🔲 Pending |
| 4 | Training & Optimization | 🔲 Pending |
| 5 | Evaluation & Testing | 🔲 Pending |
| 6 | Deployment & Demo | 🔲 Pending |

## 🔍 Key Findings (EDA)
- Dataset is balanced (~3000 images per class)
- Visually similar classes: M/N/S, U/V/R — motivates DenseNet usage
- Images are 200×200 RGB — will resize to 224×224 for DenseNet

## 🛠️ Tech Stack
- Python 3.12
- PyTorch
- DenseNet-121/169
- OpenCV, Matplotlib, Seaborn

## 📁 Project Structure

├── notebooks/ # Jupyter notebooks
├── src/ # Source code
├── models/ # Saved models
├── outputs/plots/ # EDA visualizations
└── datasets/ # (not in repo)


## 👤 Author
Anurag Singh — Galgotias College Of Engineering & Technology