import streamlit as st
from streamlit_canvas import st_canvas
import tensorflow as tf
from PIL import Image
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
    # Create canvas to draw digit
    canvas = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas"
    )
    # Make prediction when "Predict" button is clicked
    if st.button("Predict"):
        # Convert canvas to PIL image
        image = Image.fromarray(canvas.image_data.astype('uint8'), mode='L')
        # Make prediction
        label = predict(image)
        # Display predicted label
        st.write(f"Predicted label: {label}")

if __name__ == '__main__':
    app()
