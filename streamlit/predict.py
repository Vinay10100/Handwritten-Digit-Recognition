import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np

# Load TFLite model
model_path = 'streamlit/main.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image to 28x28
    image = image.resize((28, 28))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to numpy array
    image = np.array(image, dtype=np.float32)
    # Normalize image pixels to [0, 1]
    image /= 255.0
    # Expand dimensions of image to match input shape of model
    image = np.expand_dims(image, axis=0)
    # Add batch dimension to image
    image = np.stack([image], axis=0)
    return image

# Define function to make prediction
def predict(image):
    # Preprocess input image
    image = preprocess_image(image)
    # Set input tensor to preprocessed image
    interpreter.set_tensor(input_details[0]['index'], image)
    # Run inference
    interpreter.invoke()
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    # Get predicted label
    label = np.argmax(output)
    # Return predicted label
    return label

# Define Streamlit app
def app():
    st.title("Handwritten Digit Recognition")
    # Create drawing area to draw digit
    st.write("Draw a digit:")
    canvas = st.image(
        Image.new('L', (300, 300), color=255),
        width=300,
        height=300,
        caption="Draw on the white area above.",
    )
    # Make prediction when "Predict" button is clicked
    if st.button("Predict"):
        # Get image from drawing area
        image = canvas.image.copy().convert('L')
        # Make prediction
        label = predict(image)
        # Display predicted label
        st.write(f"Predicted label: {label}")

if __name__ == '__main__':
    app()
