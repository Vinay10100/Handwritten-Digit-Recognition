# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)
This project demonstrates Handwritten-Digit-Recognition using (CNN) Convolutional Neural Networks.
## Dataset
The project uses the famous MNIST dataset, which consists of 60,000 labeled images of handwritten digits for training and 10,000 labeled images for testing. Each image is 28x28 pixels in size and grayscale, with pixel values ranging from 0 to 255. The dataset is preprocessed to normalize pixel values and split into training and testing sets.
## Model Architecture
* The first layer is a convolutional layer (Conv2D) with 32 filters, a kernel size of (3, 3), and a ReLU activation function. It takes an input image of shape (28, 28, 1) where 1 represents grayscale channel.
* The output from the convolutional layer is then passed through a max pooling layer (MaxPooling2D) with a pool size of (2, 2), which helps reduce spatial dimensions while preserving important features.
* Another convolutional layer with 64 filters, a kernel size of (3, 3), and a ReLU activation function is added, followed by another max pooling layer with a pool size of (2, 2).
* The output from the last max pooling layer is then flattened into a 1D array using a Flatten layer, which prepares the data for the fully connected layers.
* Two fully connected layers (Dense) are added on top of the flattened output. The first dense layer has 128 units with a ReLU activation function, while the second dense layer has 10 units with a softmax activation function, which gives the probability of each class (0 to 9) being the correct digit.

## Model Evaluation
![1](https://user-images.githubusercontent.com/97530517/232014919-390ab15f-67e6-4a63-bef3-9005d795135f.PNG)
## Sample Output
![image (1)](https://user-images.githubusercontent.com/97530517/232014753-7cd8a16c-1b42-4a5c-b67b-27998331ef8e.png)

## How to Run

To run the Streamlit app, follow these steps:

1. Install the required dependencies, Run the Streamlit app:

   ```bash
   pip install -r requirements.txt

   streamlit run App.py
