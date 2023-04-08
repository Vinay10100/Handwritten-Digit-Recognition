import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('streamlit/main.tflite')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to a numpy array
    image = np.array(image)
    # Reshape the array to a 4D tensor with shape (1, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1)
    # Normalize the pixel values from [0, 255] to [-1, 1]
    image = (image - 127.5) / 127.5
    return image

# Define the main function
def main():
    # Set the title and description of the app
    st.title("Handwritten Digit Recognition App")
    st.write("Draw a digit from 0 to 9 on the canvas below and click the 'Predict' button to see the model's prediction.")

    # Create a canvas to draw on
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Get the image data from the canvas
    if canvas.image_data is not None:
        image = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('RGB')
        st.image(image, caption="Your drawing", use_column_width=True)

    # Add a button to predict the digit
    if st.button('Predict'):
        if canvas.image_data is None:
            st.write("Please draw a digit on the canvas above.")
        else:
            # Preprocess the image
            image = preprocess_image(image)
            # Get the model's prediction
            prediction = model.predict(image).argmax()
            # Display the prediction
            st.write("The model predicts that the digit is:", prediction)

# Run the app
if __name__ == '__main__':
    main()
