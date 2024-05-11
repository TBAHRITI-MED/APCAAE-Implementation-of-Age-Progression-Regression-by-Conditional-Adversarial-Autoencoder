# APCAAE :  Implementation of Age Progression/Regression by Conditional Adversarial Autoencoder

## System Architecture

The system architecture was written in Python 3.7 and PyTorch 0.4.1, with attempts to keep the code ascompatible as possible with older versions of Python 3 and PyTorch.
Other external packages that wereused are NumPy, scikit-learn, OpenCV, imageio and Matplotlib.

The network is comprised of an encoder which transforms RGB images to Z vectors (vectors in a latent space), a generator which transforms vectors to RGB images, a discriminator that measures (and forces) uniform distribution on the encoder's output and a discriminator that measures (and forces) realistic properties on the generator's output.

### Encoder

Encoder with 5 convolutional layers and a fully connected layer.
Viewing from left to right, faceimages of dimensions 128x128x3 are transformed into unlabeled Z vectors of size 50 in a latent space.
![alt ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/encoder.png)

### Generator

Generator with 7 deconvolutional layers and a fully connected layer.
Viewing from left to right,labeled Z vectors of size 70 in a latent space are transformed into face images of dimensions 128x128x3.
![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/Generator.png)

### Descriminators:

Discriminator on Z with 4 fully connected layers.
![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/disZ.PNG)

Discriminator on images with 4 convolutional layers and 2 fully connected layers.
![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/disImg.PNG)


### Prerequisites

* Python >= 3.5
* PyTorch >= 0.4.1
* Python data science and graphic packages: NumPy, scikit-learn, OpenCV, imageio and Matplotlib

## Training

For training, the [UTKFace dataset](http://aicip.eecs.utk.edu/wiki/UTKFace) is used, which was collected by the original authors of the article and tested in their implementation.
UTKFace contains over 20,000 aligned and cropped face images withtheir appropriate labels. A special utility was written, the UTKFace Labeler, which sorts the dataset images to separated folders based on the label, to match with PyTorch demands that classes are determined by folders structure.

Before training, one random batch of images is separated from the dataset and used for validation, meaning that the network does not back propagate losses on it.
the losses on the validation batch are expected to decrease at each epoch similarly to their change in the rest of the dataset.
After every epoch, an image comparing the original validation images with the reconstructed images is saved to the epoch's folder, allowing a human eye to monitor the training session.
An example can be seen here:

![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/validation.gif)

Original images are on the right and generated images are on the left. It can be seen that centered, frontal images with natural postures reconstruct more accurately than others.
Also, rare objects such as glasses, jewelry and watermarks are subdued.

At the end of each epoch, all of the calculated losses are passed to a class I designed, called Loss Tracker.
The loss tracker object produces graphs of the changes in losses over epochs and saves them, again to allow a human to analyze and verify the training session.
The loss tracker object also enables pre-programmed heuristics to address issues such as overfitting, underfitting, unknown fitting, and drift.
It is also possible to watch the graphs update in a new window during training. An example can be seen here:

![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/losses.gif)

To start a training session run ``` main.py --mode train <num of epochs> --input <dataset path> --output <results path>```

For the full list of options for the training session run  ``` main.py --help ```

## Applications

A few applications were developed over Jupyter Notebook to test the system with the trained models interactively.
As inputs, users can choose between already labeled images from UTKFace, to observe the results with regard to parameters such as age, gender, and race.
The applications, referred as Games, can be seen further down this section.

[The Aging Game](https://github.com/DaoudiAmir/APCAAE/blob/main/aging_game.ipynb). An input image is fed to the encoder, and the resulted Z vector is fed to the generator ten times, each time with the true gender and a different age group.
Then, the original image is presented next to all of the output images.
The output images can be seen as the aging process of a person, from childhood to old age.
The original image and the generated image of the same age group are marked in a white rectangle, for comparison.

![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/aging_game.PNG)



[The Morph Game](https://github.com/DaoudiAmir/APCAAE/blob/main/morphing_game.ipynb).
Two input images are fed to the encoder, and the resulted Z vectors are concatenated with their true labels.
Then, a set of ![](https://latex.codecogs.com/gif.latex?N&plus;1) vectors is created, with a gradual change from the first vector to the second vector, ![](https://latex.codecogs.com/gif.latex?%5C%7B%20%5Cfrac%7BN-i%7D%7BN%7D%20%5Cvec%7Bz_%7B1%7D%7D%20&plus;%20%5Cfrac%7Bi%7D%7BN%7D%20%5Cvec%7Bz_%7B2%7D%7D%20%5C%7D_%7Bi%3D0%7D%5E%7BN%7D).
The set of vectors is fed to the generator.
The output images can be seen as a morphing process from one person to another, where not only the personality features change but also age and gender, allowing to examine concepts such as immediate age transition between age groups and gender fluidity.

![alt  ](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/morph_game.PNG)



[The Kids Game](https://github.com/DaoudiAmir/APCAAE/blob/main/Tools/kids_game.ipynb).
Two input images are fed to the encoder.
Then, per each index ![](https://latex.codecogs.com/gif.latex?i) of the Z vectors, a random float ![](https://latex.codecogs.com/gif.latex?r) is generated uniformly in the semi-open range ![](https://latex.codecogs.com/gif.latex?%5B0.0%2C%201.0%29), and a new Z vector element is generated by ![](https://latex.codecogs.com/gif.latex?%5Cvec%7Bz_%7Bnew%7D%7D%5Bi%5D%20%3D%20r%20%5Ctimes%20%5Cvec%7Bz_%7B1%7D%7D%5Bi%5D%20&plus;%20%281-r%29%20%5Ctimes%20%5Cvec%7Bz_%7B2%7D%7D%5Bi%5D), so that a new Z vector ![](https://latex.codecogs.com/gif.latex?%5Cvec%7BZ_%7Bnew%7D%7D) is created and each of its features contains a mixture of the first and second images' features.