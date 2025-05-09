import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Load trained model
model = load_model('final_model.h5')
gdrive_url ="https://colab.research.google.com/drive/1j5kB08YQZkvcHZECgJjsMhu4zhzR5hvY?usp=sharing"

# Download if not already present
if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

model = load_model(model_path)
# Replace these with your actual class names
class_names = ['defect_1', 'defect_2', 'no_defect']

st.title("🔍 Metal Defect Detection")
st.write("Upload an image to detect metal defects using a trained CNN model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = load_img(uploaded_file, target_size=(256, 256))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
