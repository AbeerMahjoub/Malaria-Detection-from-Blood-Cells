import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

st.header('Malaria Detection from Blood Cells 🩸')

MODEL = tf.keras.models.load_model("models/malaria_pred_cnn.h5")

CLASS_NAMES = ['Parasite', 'Uninfected']


# helper functions
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
def predict(img):
    image = read_file_as_image(img.read())
    image = tf.image.resize(image, size=(151, 136))
    img_batch = np.expand_dims(image, 0)

    prediction = np.squeeze(MODEL.predict(img_batch))
    if prediction < .5:
        pred_class = CLASS_NAMES[0]
        confidence = np.round((1 - prediction) * 100, 2)
    else:
        pred_class = CLASS_NAMES[1]
        confidence = np.round(prediction * 100, 2)


    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = np.max(predictions[0])
    return {
        'class': pred_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    cell_img = st.file_uploader("Blood Cell Picture", label_visibility="visible", type = 'png')
    if cell_img:
        result = predict(cell_img)

        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(cell_img)

        with col2:
            st.write(f"Result: {result['class']}")
            st.write(f"Confidence: {result['confidence']} %")






