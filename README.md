# Dog vs. Cat Classification

## Overview
This project aims to classify images of dogs and cats using machine learning techniques. It utilizes the Dogs vs. Cats dataset sourced from a Kaggle competition. The goal is to build a neural network model that accurately distinguishes between images of dogs and cats.

## Dataset
The dataset used for this project consists of images of dogs and cats downloaded from the Kaggle competition "Dogs vs. Cats." It contains a total of 25,000 labeled images, with 12,500 images of dogs and 12,500 images of cats.

## Features
- **Data Loading and Preprocessing**: The Kaggle API is used to download the dataset, which is then extracted and preprocessed. Image resizing and label assignment are performed to prepare the data for training.
- **Neural Network Architecture**: Transfer learning is applied using the MobileNet V2 model as a feature extractor, followed by a dense layer with two neurons for classification.
- **Model Training**: The neural network model is trained on the resized and scaled images of dogs and cats for 5 epochs.
- **Model Evaluation**: The trained model is evaluated on the test data, achieving a test accuracy of approximately 97%.
- **Prediction**: A predictive system is built to classify new images of dogs or cats provided by the user.

## Dependencies
- NumPy
- Matplotlib
- OpenCV (cv2)
- TensorFlow
- TensorFlow Hub
- Kaggle

## Usage
1. Ensure you have the necessary dependencies installed, including the Kaggle API key for dataset download.
2. Run the provided code in a Python environment such as Jupyter Notebook or Google Colab.
3. The code will download, preprocess, train, evaluate, and make predictions using the Dogs vs. Cats dataset.
4. You can modify the code, experiment with different neural network architectures or hyperparameters, and explore further improvements in classification accuracy.
