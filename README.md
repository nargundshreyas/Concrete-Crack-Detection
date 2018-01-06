# Concrete-Crack-Detection

This repository contains the code for crack detection in concrete surfaces.

The dataset.py file creates the training dataset class to be fed into the Convolutional Neural Network. The architecture of the CNN has been taken from the following paper - "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks" by Young-Jin Cha and Wooram Choi. 

![CNN_Archi](https://user-images.githubusercontent.com/32497274/34506710-30363d94-effd-11e7-864a-bec0d7153721.PNG)

The model acheived 85% accuracy on the validation set. A few results are shown below -
![results](https://user-images.githubusercontent.com/32497274/34510394-8e4ec3e6-f021-11e7-8a70-394219f76ff2.PNG)

MATLAB was used to prepare the data. Regions of Interest were sliced into smaller 128 x 128 pixel images and used for training - 

![roi](https://user-images.githubusercontent.com/32497274/34510417-c3207466-f021-11e7-9bf7-c91c034a70be.PNG)
