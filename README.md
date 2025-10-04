# Image Classification of Dry Fruits

This project demonstrates the use of deep learning to classify images of different types of dry fruits. It explores two different approaches: a simple Convolutional Neural Network (CNN) built from scratch and a more advanced approach using transfer learning with the MobileNetV2 architecture.

## Dataset

The project uses the [Dry Fruit Image Dataset](https://data.mendeley.com/datasets/yfhgn8py5f/1), which contains images of 12 different types of dry fruits. The dataset is organized into folders, with each folder representing a different class of dry fruit.

## Models

Two models are implemented in this project:

### 1. Simple CNN

A simple CNN is built from scratch to serve as a baseline model. The architecture consists of several convolutional and max-pooling layers, followed by a dense layer for classification.

### 2. Transfer Learning with MobileNetV2

This model uses the MobileNetV2 architecture, pre-trained on the ImageNet dataset, as a feature extractor. The top layers of the model are replaced with a new set of layers that are trained on the dry fruit dataset.

## Getting Started

To run this project, you will need to install the dependencies from the `requirement.txt` file.

You can install the dependencies by running the following command in your terminal:

```bash
pip install -r requirement.txt
```

Once you have the required libraries installed, you can run the `notebook1.ipynb` file in a Jupyter Notebook environment.

## Results

The results of the two models are compared based on their accuracy and loss on the validation set. The transfer learning model is expected to outperform the simple CNN, as it leverages the knowledge learned from the ImageNet dataset.

The notebook includes visualizations of the training and validation accuracy and loss for both models, as well as a comparison of their predictions on a batch of validation images.
