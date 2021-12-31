[//]: # (Image References)

[image1]: ./Test_Images/Expected_Output.jpg "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

This project is one of the suggested capstone projects in the Data Scientist Nanodegree. In this project I have build the Convolutional Nueral Network that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

### Example Results

![Sample Output][image1]

The code is written in Python 3 and Keras with Tensorflow backend all presented in Jupyter Notebook.
### Project Requirements

This project requires Python and libraries used in Data Science and Deep Learning such as:

- numpy
- pandas
- matplotlib
- scikit-learn
- keras

To speed up training time, it is highly recommended to use GPU. The dataset used for this project can be downloaded from the following links:
- Dog Dataset: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 
- Human Dataset: [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`. 
- VGG-16 botthleneck features: The [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

### Blog Post

This repository is discussed in the blog post here.

### Licensing, Authors, Acknowledgements

The dataset and the initial workspace was provided by Udacity as part of its Data Scientist Nanodegree program. A LICENSE file is added in this repo to state clearly its licence.
