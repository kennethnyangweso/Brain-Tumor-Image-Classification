import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    
    # Modify classifier if you changed it during training
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # Change 3 to your number of classes
    
    model.load_state_dict(torch.load("efficientnet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names (EDIT THIS)
class_names = ["giloma", "meningioma", "notumor", "pituitary"]  # Change to your actual class names

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# UI
st.title("🧠 EfficientNet Image Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    st.success(f"Prediction: {class_names[predicted.item()]}")
    st.info(f"Confidence: {confidence.item():.2f}")