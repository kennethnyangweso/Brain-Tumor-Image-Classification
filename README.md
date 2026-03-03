<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f17581b6-6da8-46fe-bbe3-dc3354f15e9e" />

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

The brain classes include:

1️⃣ Glioma - A tumor that occurs in the brain and spinal cord, originating from glial cells.
Often aggressive and requires early detection.

2️⃣ Meningioma - A tumor that forms on membranes covering the brain and spinal cord.
Typically slow-growing and often benign.

3️⃣ Pituitary Tumor - A tumor located in the pituitary gland at the base of the brain.
Can affect hormone production and bodily functions.

4️⃣ No Tumor - MRI scan shows no visible tumor presence. Represents healthy brain imaging.

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

| METRIC/MODEL | ResNet-50 | EfficientNet-B0 | DenseNet-121 |
|--------------|-----------|----------------|--------------|
| ACCURACY (%) | 87.60     | 98.84          | 98.57        |
| RECALL (%)   | 87.74     | 98.84          | 98.58        |
| F1-SCORE (%) | 87.48     | 98.84          | 98.57        |

<img width="975" height="582" alt="image" src="https://github.com/user-attachments/assets/86df5d75-1823-4cfd-9575-cc1224411f9a" />

### **Key Insights**

- Best performing model: EfficientNet-B0 (slightly better than DenseNet-121 and more efficient).

-	ResNet-50: Adequate but significantly worse, might not be suitable for high-stakes deployment.

- 	Balanced performance: All top two models have high recall and F1-score, meaning they handle class distribution well and are reliable in predicting the target.

<img width="975" height="772" alt="image" src="https://github.com/user-attachments/assets/5494e668-9603-461b-9b8a-6006dc06a52e" />

#### **Model Comparison via ROC Curves**

-	ResNet-18 (blue dashed line) has a slightly lower ROC curve, with AUC = 0.981.

-	EfficientNet-B0 (orange solid line) achieves a perfect ROC curve with AUC = 1.000.

-	DenseNet-121 (green dash-dot line) also performs extremely well with AUC = 0.999, almost indistinguishable from EfficientNet-B0.

### **Best Model (EfficientNet) Confusion Matrix:**

<img width="975" height="727" alt="image" src="https://github.com/user-attachments/assets/bf83d093-58ad-4a6a-8a15-4d7cfd9fe669" />

#### **Insights**

**Most Confusions:**
- Meningioma vs Glioma (2 cases misclassified)
-	Meningioma vs Pituitary (3 cases misclassified)
-	These are very few, showing the model captures features of each tumor type well.

**Overall Accuracy:** 
- Extremely high; very few errors across ~1140+ samples in total.
- Class-wise Sensitivity:
- All classes have high recall, with Glioma, NoTumor, and Pituitary almost perfect.
- Meningioma is slightly more confused but still >98% correct.

---
## 🚀 **Deployment/App**

### 📂 **Project Structure**

```│
├── app.py # Streamlit app
├── efficientnet_model.pth # Trained model weights
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── data/ #  dataset folder
└── assets/ #  screenshots, diagrams
```

-	Framework: Streamlit
-	Model Saved: efficientnet_model.pth (PyTorch)

Features:
-	Upload MRI image for prediction
-	Shows predicted tumor class and confidence
-	Probability bar chart visualization

Deployment Options:

- Streamlit Community Cloud (live link possible)
-	Local run via
  
```streamlit run app.py```

This allows stakeholders to interactively test the model on new images.

#### **Expected Results**

<img width="1023" height="614" alt="image" src="https://github.com/user-attachments/assets/5c134a2b-fbc5-471c-890e-ffaff8c94a8c" />

<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/e9b633ce-9541-417a-8aef-57336dcfb5ba" />

<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/a4d908ea-92cc-4cd6-9abd-2393d8a78c3e" />

Video demonstration link (copy the link below to view live demonstration of the brain tumor classifier):

Link: https://drive.google.com/file/d/1KZSMKYDckz_CbDMhihBngb5glYVnoFJc/view?usp=drive_link

---

## **💡 Key Insights / Business Recommendations**

-	EfficientNet-B0 demonstrated outstanding performance on brain MRI classification.
-	Deep learning can automate tumor detection, reduce radiologist workload, and increase diagnostic speed.
-	Expand dataset with more diverse MRI images for robustness.
-	Apply data augmentation to further improve generalization.
-	Integrate Grad-CAM or attention maps for explainability in clinical settings.
-	Deploy as a web service for real-time hospital use.
- Monitor model performance with new incoming data and retrain periodically

---

## 📝 **Installation**

```bash
git clone https://github.com/kennethnyangweso/Brain-Tumor-Image-Classification.git
cd brain-tumor-classification
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## **👤 Author**

**Kenneth Nyangweso**

**Data Scientist | AI Engineer**
