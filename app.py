import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# STEP 1: Download the model from Google Drive if not already downloaded
file_id = '1ulXy2N4-ofhXI9i5hd_TvF53WClJhzkf'  # Ensure this is the correct file ID
url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'final_model.h5'

# Check if model already exists
if not os.path.exists(model_path):
    st.info("🔽 Downloading model from Google Drive...")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        st.stop()  # Stop execution if download fails

# STEP 2: Verify if the file is valid
if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
    try:
        model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()  # Stop execution if model loading fails
else:
    st.error("❌ Model download failed or file is incomplete.")
    st.stop()  # Stop execution if model is invalid

# STEP 3: Define class names (update with your actual defect classes)
class_names = ['defect_1', 'defect_2', 'no_defect']

# STEP 4: Build Streamlit interface
st.title("🔍 Metal Defect Detection")
st.write("Upload an image to detect metal defects using a trained CNN model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = load_img(uploaded_file, target_size=(256, 256))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
