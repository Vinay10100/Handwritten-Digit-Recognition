import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.title('Handwritten Digit Recognition')
st.subheader("Draw the digit on canvas and click on 'Predict Now'")

# Add canvas component
# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = st.slider('Select Stroke Width', 1, 30, 15)
stroke_color = '#FFFFFF' # Set background color to white
bg_color = '#000000'

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Add "Predict Now" button
if st.button('Predict Now'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('Predicted Digit: ' + str(res))
    else:
        st.header('Please draw a digit on the canvas.')

# Add sidebar
st.sidebar.title("About")
st.sidebar.text("Created by Vinay Uniyal")
st.sidebar.text("GitHub Repository")
st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
