import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Handwritten Digit Recognition App")

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mnist.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    # Preprocess the image.
    image = np.array(image)
    image = image[:,:,3]
    image = Image.fromarray((image).astype(np.uint8))
    image = image.resize((28,28))
    image = ImageOps.invert(image)
    image = np.array(image)
    image = image.reshape(1,28,28,1)
    image = image.astype('float32')
    image /= 255
    
    # Run inference on the model.
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(pred)

def main():
    st.write("Draw a digit from 0 to 9")
    canvas = st.sketchpad(width=200,height=200)
    if st.button("Predict"):
        digit = predict(canvas.image_data)
        st.write("Prediction:", digit)

if __name__ == "__main__":
    main()
