
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import time

st.set_page_config(
    page_title="Cassava Leaf Disease Screening",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Cassava Leaf Disease Screening System")
st.write(
    "Upload a cassava leaf image to screen it for possible bacterial blight, "
    "healthy condition, or mosaic disease."
)

MODEL_PATH = "cassava_mobilenetv2_finetuned_model.keras"

CLASS_NAMES = [
    "Bacterial Blight",
    "Healthy",
    "Mosaic Disease"
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a cassava leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded cassava leaf image",
        use_container_width=True
    )

    if st.button("Analyse Leaf"):
        with st.spinner("Analysing leaf image..."):
            start_time = time.time()

            img = image.resize((224, 224))
            img_array = np.array(img).astype("float32")
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = np.max(prediction) * 100
            response_time = time.time() - start_time

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.write(f"Response time: {response_time:.2f} seconds")

        if predicted_class != "Healthy":
            st.warning(
                "Possible disease detected. Please consult an agricultural "
                "extension officer or plant-health expert for confirmation."
            )
