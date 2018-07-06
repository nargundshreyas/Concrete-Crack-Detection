# Concrete-Crack-Detection

This repository contains the code for crack detection in concrete surfaces. It is a TensorFlow implementation of the paper by by Young-Jin Cha and Wooram Choi - "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks".

![CNN_Archi](https://user-images.githubusercontent.com/32497274/34506710-30363d94-effd-11e7-864a-bec0d7153721.PNG)

The model acheived 85% accuracy on the validation set. A few results are shown below -
![results](https://user-images.githubusercontent.com/32497274/34510394-8e4ec3e6-f021-11e7-8a70-394219f76ff2.PNG)

MATLAB was used to prepare the data. Regions of Interest were sliced into smaller 128 x 128 pixel images and used for training - 

![roi](https://user-images.githubusercontent.com/32497274/34510417-c3207466-f021-11e7-9bf7-c91c034a70be.PNG)

Dependencies required-<br />
- TensorFlow<br />
- OpenCV

The dataset.py file creates the training dataset class to be fed into the Convolutional Neural Network. This class automatically determines the number of classes by the number of folders in 'in_dir' (number of folders=number of classes)


NOTE: This script utilizes an already saved cache file. The cache file contains the filenames of the data. If the cache file doesn't exist, a new one created. If the data has been altered with, please delete the old cache file and run the script again. This applies for both `Running.py` and `Train_CD.py`. This was my first ever Deep Learning project hence, the naive approach. I'll streamline this once I get some time, or gladly accept a pull request!

The directory structure is assumed to be the following- (For example considering 3 classes)<br />
* in_dir/class1/              - Contains all the training images for class 1<br />
    * test/         - Contains all the validation images for class 1 <br />
* in_dir/class2/              - Contains all the training images for class 2<br />
    * test/         - Contains all the validation images for class 2<br />
* in_dir/class3/              - Contains all the training images for class 3<br />
    * test/         - Contains all the validation images for class 3<br />
  
To train the network run the command with the following arguments:<br />
`python Train_CD.py`<br />

Argument | Details | Default
--- | --- | --- 
`--in_dir` | path to *in_dir* folder | *cracky* 
`--iter` | number of iterations to run the model for | 1500 
`--save_folder` | Directory to save checkpoint | CURRENT_DIR 

After model has been trained, meta_files are saved into 'save_folder'. To test the model, run the command with the following arguments:
`python Running.py` 

Argument | Details | Default
--- | --- | --- 
`--in_dir` | directory containing unlabeled test data |*cracky_test* 
`--meta_file` | MetaFile path | None (Will throw error if not given)
`--CP_dir` | dir contatining checkpoint | None (Will throw error if not given)
`--save_dir` | dir to save output images | CURRENT_DIR  

## TODO:

 - [ ] Streamline data loading; remove cache file system
 - [ ] Combine training and testing scripts into one
