import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('/mount/src/cpe019_final_-exam/NasNet_model.h5')

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

st.title("Weather Image Classifier (NASNet)")

uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    st.write(f"**Prediction:** {class_names[class_idx]}")
