import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import base64
from huggingface_hub import snapshot_download




# Set background image using local file
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("Satellite.jpg")

st.title("Satellite Image Classification")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    model_dir = snapshot_download(
        repo_id="Tusharyadav/satellite-image-classifier",
        repo_type="model"
    )
    saved_model_path = os.path.join(model_dir, "satellite_cnn_savedmodel")
    model = tf.keras.models.load_model(saved_model_path)
    return model

class_names = [
     "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def classify(model, image: Image.Image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    class_label = class_names[predicted_index]
    return class_label, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    model = load_model()
    label, confidence = classify(model, image)
    st.success(f"Prediction: {label} ({confidence:.2%} confidence)")
