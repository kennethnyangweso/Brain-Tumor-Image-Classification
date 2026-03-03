import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detector")
st.markdown("Deep Learning model using EfficientNet (PyTorch)")
st.write("Upload an MRI image to classify the tumor type.")

# -----------------------------
# Class Names
# -----------------------------
class_names = ["giloma", "meningioma", "notumor", "pituitary"]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    
    # Modify classifier to match 4 classes
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 
        4
    )
    
    model.load_state_dict(
        torch.load("efficientnet_model.pth", map_location=torch.device("cpu"))
    )
    
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    st.success(f"Prediction: {predicted_class.upper()}")
    st.info(f"Confidence: {confidence_score:.4f}")

    # -----------------------------
    # Probability Breakdown Chart
    # -----------------------------
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probabilities.numpy().flatten()
    })

    st.subheader("Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Class"))