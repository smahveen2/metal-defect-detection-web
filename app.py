import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')

st.title("🔎 Metal Surface Defect Detector")

uploaded_file = st.file_uploader("Upload a metal surface image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (resize to match your model's input size)
    img = image.resize((224, 224))  # Change this to your model's expected size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)

    st.write("Prediction:", prediction)

    # Optional: Map prediction to labels
    if prediction[0][0] > 0.5:
        st.success("✅ Defect Detected!")
    else:
        st.info("🟢 No Defect Detected!")
