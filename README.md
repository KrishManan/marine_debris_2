# Marine Debris Classification Project

## Overview

This project aims to develop a neural network model to classify images of marine debris based on their danger level. The approach involves using a YOLOv10 network for object detection to identify marine debris in images, followed by using a ResNet50 model to classify the danger level of the detected debris.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Structure

marine-debris-classification/ │ ├── data/ # Dataset directory │ ├── train/ # Training images │ ├── val/ # Validation images │ └── annotations/ # Annotations for training │ ├── models/ # Directory for model definitions │ ├── yolov10/ # YOLOv10 model │ └── resnet50/ # ResNet50 model │ ├── utils/ # Utility functions │ ├── results/ # Directory for saving results and logs │ ├── train.py # Main script for training ├── evaluate.py # Script for evaluating models └── README.md # Project documentation


## Getting Started

To get started with this project, clone the repository and install the required dependencies. You can do this using the following commands:

```bash
git clone https://github.com/yourusername/marine-debris-classification.git
cd marine-debris-classification
pip install -r requirements.txt

##Requirements
Python 3.x

PyTorch

torchvision

OpenCV

Other required libraries listed in requirements.txt

##Model Architecture
YOLOv10: This model is utilized for detecting marine debris in images. It outputs bounding boxes and class labels for the identified debris.

ResNet50: After detecting the debris, the images (with bounding boxes) are fed into a ResNet50 model to classify the danger level of the debris on a scale from 1 to 3.

##Dataset
The dataset consists of images containing various types of marine debris, annotated with bounding boxes and classes. The dataset is split into training and validation sets, with annotations provided in a compatible format for the YOLO model. 

##Dataset Sources
Some images taken and generated by ourselves
Others from this open source dataset https://universe.roboflow.com/reconhecimentoimgs/global-solution/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

Training
To train the models, execute the following command:

python train.py --dataset path_to_your_dataset --epochs 50 --batch_size 16
Adjust the parameters based on your needs.

Evaluation
To evaluate the trained models on the validation set, use the following command:

python evaluate.py --model_path path_to_trained_model --dataset path_to_validation_dataset
Results
The results of the model training and evaluation will be saved in the results/ directory. You can visualize the results using TensorBoard or any other visualization tool of your choice.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
YOLOv10 for object detection.

ResNet for image classification.

Any other relevant acknowledgments or references.

Feel free to contribute to the project, report issues, or suggest improvements!
