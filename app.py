import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from pyngrok import ngrok

public_url = ngrok.connect(8501)
print("Public URL:", public_url)

# создаём ту же модель
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# загружаем веса
model.load_weights("emotion_model_rafdb.h5")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

st.title("Emotion AI 🧠")
st.write("Upload a face image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        face = cv2.resize(img, (224,224))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]

        emotion_id = np.argmax(preds)
        emotion = emotion_labels[emotion_id]
        confidence = preds[emotion_id] * 100

        st.success(f"Emotion: {emotion} ({confidence:.2f}%)")

    except Exception as e:
        st.error("Error processing image")