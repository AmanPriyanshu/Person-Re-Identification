# Introduction:

A Siamese Neural Network is a class of neural network architectures that contain two or more identical sub networks. Parameter updating is mirrored across both sub networks. It is used to find the similarity of the inputs by comparing its feature vectors.
Simple Siamese Network for proximity based embedding of similar images. It utilizes the triplet loss function for training.

Reasons to Use Siamese Neural Network : 
* Needs less training Examples to classify images because of One-Shot Learning
* Learn by Embedding of the image so that it can learn Semantic Similarity
* It helps in ensemble to give the best classifiers because of its correlation properties.
* Mainly used for originality verification.

## Feature-Extractor Module:

The Siamese Network consists of two modules for feature extraction, however both these modules are identical in nature. Therefore, we summarize the parameters and confidurations of this network.

1. Conv2D --> 3 channels --> 5 channels
2. ReLU
3. LocalResponseNorm
4. MaxPool2d

5. Conv2D --> 5 channels --> 10 channels
6. ReLU
7. LocalResponseNorm
8. MaxPool2d

9. Conv2D --> 10 channels --> 5 channels
10. ReLU
11. Conv2D --> 5 channels --> 3 channels
12. ReLU
13. MaxPool2d

## Fully Connected Layers:

1. Linear
2. ReLU
3. Linear
4. ReLU
5. Linear

## 