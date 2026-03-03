# 🧠 Brain Tumor Classification Using Deep Learning
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange?logo=streamlit&logoColor=white)](https://streamlit.io/) 
[![ResNet-50](https://img.shields.io/badge/ResNet--50-CC0000?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) 
[![EfficientNet-B0](https://img.shields.io/badge/EfficientNet-B0-0066CC?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/) 
[![DenseNet-121](https://img.shields.io/badge/DenseNet-121-FF6600?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) 
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF00FF?style=flat&logo=Keras&logoColor=white)](https://www.deeplearning.ai/) 
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=Seaborn&logoColor=white)](https://seaborn.pydata.org/) 
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=Python&logoColor=white)](https://matplotlib.org/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) 
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

---

## **🎯 Business Understanding:**

Medical imaging plays a critical role in diagnosing brain tumors. Rapid and accurate
classification of MRI images into tumor types can support radiologists and clinicians in making
informed treatment decisions. The goal of this project is to leverage deep learning models to
automatically classify brain MRI images into different tumor categories.
By deploying the model as a web application, hospitals and research institutions can have a
scalable, accessible tool for aiding diagnosis.

## 📌 **Project Objectives:**

This project focuses on **classifying brain MRI images** into four categories:

- `giloma`  
- `meningioma`  
- `notumor`  
- `pituitary`  

using **deep learning models** (ResNet-50, EfficientNet-B0, DenseNet-121) implemented in PyTorch and deployed as a web application using Streamlit.

The goal is to provide a fast, reliable, and interactive tool for tumor classification.

---

## 🎯 Features

- Upload MRI images for automatic classification  
- Prediction includes class and confidence score  
- Probability bar chart visualization  
- Comparison of multiple deep learning models  
- Interactive web deployment using Streamlit

---

## 🧠 Models Used

| Model          | Accuracy | Notes |
|----------------|---------|-------|
| ResNet-50      | 87.6%   | Baseline model, good for comparison |
| EfficientNet-B0| 98.8%   | Best performing, used for deployment |
| DenseNet-121   | 98.6%   | Strong alternative, slightly lower accuracy |

---

## 📂 Project Structure

```│
├── app.py # Streamlit app
├── efficientnet_model.pth # Trained model weights
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── data/ # Optional: dataset folder
└── assets/ # Optional: screenshots, diagrams
```

