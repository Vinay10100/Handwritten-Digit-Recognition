# importing the required libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# function to load the model
def load_model():
    model = tf.keras.models.load_model('streamlit/main.tflite')
    return model

# function to predict the digit
def predict(model, image):
    image = np.array(image.convert('L').resize((28,28)))
    image = image.reshape((1, 28, 28, 1))
    image = image / 255.0
    pred = model.predict(image)
    return np.argmax(pred)

# loading the model
model = load_model()

# creating a canvas for drawing the digit
canvas = st.canvas(width=150, height=150)

# predict the digit when the "Predict" button is clicked
if st.button('Predict'):
    # get the canvas image and predict the digit
    image = Image.fromarray(canvas.image_data.astype('uint8'), mode='L')
    digit = predict(model, image)
    st.write('Predicted Digit:', digit)
