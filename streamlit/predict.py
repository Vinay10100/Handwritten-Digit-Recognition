import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# global variable to hold the TensorFlow model
model = None

# function to load the model
def load_model():
    global model
    model_path = 'streamlit/main.tflite'
    model = tf.keras.models.load_model(model_path)

# function to preprocess the canvas image
def preprocess(image):
    # resize the image to 28x28 pixels and invert the colors
    image = ImageOps.invert(image.resize((28,28)))
    # convert the image to grayscale and normalize the pixel values
    image = ImageOps.grayscale(image).convert('L')
    image_array = np.array(image) / 255.0
    # reshape the array to have a single channel (for grayscale images)
    image_array = image_array.reshape((1, 28, 28, 1))
    return image_array

# function to predict the digit from the canvas image
def predict_digit(image_array):
    # predict the digit using the loaded model
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

# main Streamlit code
if __name__ == '__main__':
    # load the model
    load_model()
    
    # set up the canvas
    st.write("Draw a digit in the canvas below:")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    # make a prediction on the drawn digit when the 'Predict' button is clicked
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # get the canvas image data and preprocess it
            canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
            processed_image = preprocess(canvas_image)
            
            # predict the digit and display the result
            predicted_digit = predict_digit(processed_image)
            st.write(f"Predicted digit: {predicted_digit}")
        else:
            st.write("Please draw a digit first.")
