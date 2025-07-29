import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

st.set_page_config(page_title="Satellite Image Classifier", layout="centered")

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

st.title("Satellite Image Classifier")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])
# EuroSAT and EuroSATallBands class names as per official dataset
class_names = [
        
    "Forest",        # 1
    "HerbaceousVegetation", #
    "Highway",       # 
    "Industrial",    # 
    "Pasture",       # 
    "PermanentCrop", # 
    "Residential",   
    "River",         # 
    "SeaLake"        
]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
Predict = st.button("Classify Image")
if Predict:
  if uploaded_file is not None:
   

    # Load model
    model = tf.keras.models.load_model("satellite.h5" , compile=False)

    # Preprocess image (adjust as per your model's requirements)
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_name = class_names[class_idx]
    st.success(f"Classification Result : {class_name}")
  else:
    st.warning("Please upload an image to classify.")
        










   
