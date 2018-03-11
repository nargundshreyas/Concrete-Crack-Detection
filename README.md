# Concrete-Crack-Detection

This repository contains the code for crack detection in concrete surfaces. It is a TensorFlow implementation of the paper by by Young-Jin Cha and Wooram Choi - "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks".

![CNN_Archi](https://user-images.githubusercontent.com/32497274/34506710-30363d94-effd-11e7-864a-bec0d7153721.PNG)

The model acheived 85% accuracy on the validation set. A few results are shown below -
![results](https://user-images.githubusercontent.com/32497274/34510394-8e4ec3e6-f021-11e7-8a70-394219f76ff2.PNG)

MATLAB was used to prepare the data. Regions of Interest were sliced into smaller 128 x 128 pixel images and used for training - 

![roi](https://user-images.githubusercontent.com/32497274/34510417-c3207466-f021-11e7-9bf7-c91c034a70be.PNG)


The dataset.py file creates the training dataset class to be fed into the Convolutional Neural Network. This class automatically determines the number of classes by the number of folders in 'in_dir' (number of folders=number of classes)

The directory structure is assumed to be the following- (For example considering 3 classes)
        in_dir/class1/              - Contains all the training images for class 1
        in_dir/class2/              - Contains all the training images for class 2
        in_dir/class3/              - Contains all the training images for class 3
        in_dir/class1/test/         - Contains all the validation images for class 1
        in_dir/class2/test/         - Contains all the validation images for class 2
        in_dir/class3/test/         - Contains all the validation images for class 3
  
To train the network run the command with the following arguments:
python Train_CD.py 	--in_dir=DIRECTORY_NAME	(directory containing training data)<br />
			--iter=NUMBER_OF_ITERATIONS (number of training iterations. Default-1500)
			--save_folder=DIRECTORY_NAME (directory to save meta files and checkpoints)

After model has been trained, meta_files are saved into 'save_folder'. To test the model, run the command with the following arguments:
python Running.py 	--in_dir=DIRECTORY_NAME	(directory containing unlabeled test data)
					--meta_file=LOCATION_OF_META_FILE (path for meta_file)
					--CP_dir=DIRECTORY_NAME (directory containing checkpoints)
