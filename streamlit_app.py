import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Rebuild model architecture
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable=false
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('nasnet_model.weights.h5')
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

st.title("Weather Image Classifier (NASNet)")

uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    st.write(f"**Prediction:** {class_names[class_idx]}")
