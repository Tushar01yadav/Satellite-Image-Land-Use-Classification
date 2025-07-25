import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Class labels for EuroSAT RGB dataset
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Load a pre-trained model (replace with your own model path if needed)
@st.cache_resource
def load_model():
    # For demonstration, using a simple MobileNetV2 pretrained on ImageNet and fine-tuned for 10 classes
    # In practice, load your own trained model
    model = tf.keras.applications.MobileNetV2(
        input_shape=(64, 64, 3), weights=None, classes=10
    )
    # model.load_weights('your_trained_model.h5')  # Uncomment and set your model path
    return model

model = load_model()

st.title("Satellite Image Land Use Classification 🌍")
st.write(
    "Upload a satellite image (EuroSAT RGB) and the model will classify its land use category."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]

    st.subheader("Prediction")
    st.write(f"**Predicted Land Use:** {pred_class}")
else:
    st.info("Please upload a satellite image to classify.")
