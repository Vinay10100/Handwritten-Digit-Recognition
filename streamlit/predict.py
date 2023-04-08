import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('/streamlit/main.tflite')

# Define function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Invert the colors
    image = ImageOps.invert(image)
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array / 255.0
    # Reshape to (1, 28, 28, 1)
    image_array = np.reshape(image_array, (1, 28, 28, 1))
    return image_array

# Define the Streamlit app
def app():
    # Set the page title
    st.set_page_config(page_title='Handwritten Digit Recognizer')
    # Add a title and description
    st.title('Handwritten Digit Recognizer')
    st.write('Draw a digit in the canvas below and click the "Predict" button to see the model prediction.')
    # Create a canvas
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=200,
        width=200,
        drawing_mode='freedraw',
        key='canvas'
    )
    # Add a button to predict the digit
    if st.button('Predict'):
        # Get the canvas image data
        image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        # Preprocess the image
        image_array = preprocess_image(image)
        # Make the prediction
        prediction = model.predict(image_array)
        # Get the predicted class label
        predicted_class = np.argmax(prediction)
        # Show the prediction result
        st.write(f'Predicted digit: {predicted_class}')

# Run the app
if __name__ == '__main__':
    app()
