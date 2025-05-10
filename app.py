import os
import gdown
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# STEP 1: Download the model from Google Drive if not already downloaded
file_id = '1KdnDz466Mes5XtREb2SFTSFOdzxO_xq_'
url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'final_model.h5'

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# STEP 2: Load the model
model = load_model(model_path)

# STEP 3: Define class names
class_names = ['defect_1', 'defect_2', 'no_defect']

# STEP 4: Streamlit UI
st.title("🔍 Metal Defect Detection")
st.write("Upload an image to detect metal defects using a trained CNN model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = load_img(uploaded_file, target_size=(256, 256), color_mode='rgb')
        img_array = img_to_array(image) / 255.0

        if img_array.shape != (256, 256, 3):
            st.error("Uploaded image is not in RGB format or has the wrong size.")
        else:
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index] * 100

            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

            # Bar chart of class probabilities
            st.subheader("🔢 Class Probabilities:")
            fig, ax = plt.subplots()
            ax.bar(class_names, prediction[0] * 100, color='skyblue')
            ax.set_ylabel('Confidence (%)')
            ax.set_ylim([0, 100])
            ax.set_title("Prediction Confidence for Each Class")
            st.pyplot(fig)

    except Exception as e:
        st.error("⚠️ Error processing the image or making prediction.")
        st.exception(e)
