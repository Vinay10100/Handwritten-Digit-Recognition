from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import streamlit as st
from streamlit_canvas import st_canvas
from PIL import ImageOps, Image
import io

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="streamlit/main.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_digit(image):
    # Preprocess the image
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = image.reshape((1, 28, 28, 1))
    image = np.array(image) / 255.0

    # Make a prediction using the TFLite model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    
    return prediction


def main():
    st.title("Handwritten Digit Classification Web App")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    activities = ["Draw", "Upload"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Draw":
        st.subheader("Draw Digit")
        canvas_result = st_canvas( 
            stroke_width=20, 
            stroke_color="#fff", 
            background_color="#000", 
            height=150, 
            width=150, 
            drawing_mode="freedraw", 
            key="canvas",
        )
        if canvas_result.image_data is not None:
            img = ImageOps.invert(Image.fromarray(canvas_result.image_data.astype('uint8'), 'L').resize((28, 28)))
            st.image(img)
            if st.button("Predict"):
                digit = predict_digit(img)
                st.write("Predicted digit:", digit)

    elif choice == "Upload":
        st.subheader("Upload Image")
        file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

        if file is not None:
            image = Image.open(file).convert("L")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict"):
                digit = predict_digit(image)
                st.write("Predicted digit:", digit)


if __name__ == "__main__":
    main()

