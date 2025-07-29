import streamlit as st
from PIL import Image
import requests
import numpy as np
import os
import tensorflow as tf
from huggingface_hub import snapshot_download
import base64


# Set background image using CSS
def get_base64_of_local_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_of_local_image("Satellite.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Satellite Image Classification")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    # Download the entire model repo from Hugging Face
    model_dir = snapshot_download(
        repo_id="Tusharyadav/satellite-image-classifier",
        repo_type="model"
    )

    # Path to your saved model folder inside the repo
    saved_model_path = os.path.join(model_dir, "satellite_cnn_savedmodel")

    # Load the TensorFlow model
    model = tf.keras.models.load_model(saved_model_path)
    return model
class_names = [
     "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def classify(model, image: Image.Image):

    # Resize and normalize the image
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    # Get class label
    class_label = class_names[predicted_index]
    return class_label, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Replace with actual model URL when provided
    
    model = load_model()
    label, confidence = classify(model, image)
    st.success(f"Prediction: {label} ({confidence:.2%} confidence)")
