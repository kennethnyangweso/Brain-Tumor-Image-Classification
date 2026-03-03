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

1. Develop deep learning models capable of classifying brain MRI images.
2. Compare multiple models to identify the most effective for this task.
3. Deploy the best performing model as an interactive web application for demonstration.
4. Provide insights and performance metrics for future improvements.

---

## 🎯 Features

- Upload MRI images for automatic classification  
- Prediction includes class and confidence score  
- Probability bar chart visualization  
- Comparison of multiple deep learning models  
- Interactive web deployment using Streamlit

---

## **🗂️ Data Overview**

-  Dataset: Brain MRI images from publicly available medical imaging datasets.
-  Classes: glioma, meningioma, no tumor, pituitary.
-  Training set: 5600 images across all classes.
-  Validation set: 1120 images across all classes.
-  Image Size: 224x224 pixels after preprocessing for model input.

Each class has a sufficient number of images to enable robust model training.

## 🧠 Modeling

Three transfer learning models were trained and deployed :

-  RestNet-50
-  EfficientNet-B0
-  DenseNet121

**Implementation Details:**

- Framework: PyTorch
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Training: GPU accelerated (T4 GPU)
- Epochs: 10–20 depending on model
- Evaluation metrics included confusion matrices, classification reports, and ROC curves.

---

## 🧩 **Evaluation**



## 📂 Project Structure

```│
├── app.py # Streamlit app
├── efficientnet_model.pth # Trained model weights
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── data/ #  dataset folder
└── assets/ #  screenshots, diagrams
```

