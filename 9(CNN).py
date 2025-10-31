Theory: Convolutional Neural Network (CNN) for Handwritten Digit Recognition
1. Introduction
Handwritten digit recognition is one of the most common and fundamental applications of deep learning.
It involves automatically identifying digits (0–9) from handwritten images.
In this experiment, we use a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset, which contains 70,000 grayscale images of digits written by different people.

2. What is a CNN?
A Convolutional Neural Network (CNN) is a special type of neural network designed to process image data.
Unlike a fully connected neural network, CNNs automatically detect spatial patterns such as edges, corners, and textures, making them ideal for image classification and recognition tasks.

A typical CNN architecture has the following layers:
(a) Convolution Layer
Extracts features from the input image using a set of learnable filters (kernels).
Each filter detects specific features such as lines, edges, or corners.
Output of this layer is called a feature map.

(b) Activation Function (ReLU)
ReLU (Rectified Linear Unit) introduces non-linearity in the network.
It helps the model learn complex relationships and speeds up training.

(c) Pooling Layer
Reduces the spatial size of feature maps while retaining important information.
Common types: Max Pooling and Average Pooling.
This layer helps reduce computation and prevent overfitting.

(d) Flatten Layer
Converts the 2D feature maps into a 1D vector before passing them to dense (fully connected) layers.

(e) Fully Connected (Dense) Layer
Performs classification based on the extracted features.
Each neuron connects to all activations in the previous layer.

(f) Output Layer
Uses a Softmax activation function to produce probability scores for each class (digit 0–9).

3. Dataset Used – MNIST
Full form: Modified National Institute of Standards and Technology dataset
Images: 70,000 grayscale images (60,000 for training, 10,000 for testing)
Image Size: 28 × 28 pixels
Classes: 10 (digits 0 to 9)
Each pixel value: Ranges from 0 (black) to 255 (white)

4. Libraries Used
Library	Purpose
TensorFlow / Keras	To build and train the CNN model
NumPy	Numerical operations and array handling
Matplotlib	Visualization of digits and predictions
                                                                
5. Working of the Model
Input Layer: Takes 28×28 grayscale image.

Convolution + Pooling Layers: Automatically extract key features such as curves and edges.
Flatten Layer: Converts feature maps into a vector.
Dense Layer: Learns complex relationships between features.
Output Layer: Predicts the class (digit) using Softmax activation.
The model is trained using the Adam optimizer and Sparse Categorical Cross-Entropy loss, and accuracy is measured on both training and testing datasets.

6. Results
After training for around 10–12 epochs, the CNN achieves an accuracy of about 98–99% on the MNIST test dataset.
The model successfully classifies most handwritten digits correctly.

7. Conclusion
In this experiment, we successfully implemented a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset.
The CNN automatically learned visual features without manual extraction and provided high classification accuracy.
This demonstrates that CNNs are powerful models for image recognition and computer vision tasks such as handwritten digit classification.
