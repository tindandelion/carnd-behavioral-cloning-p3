# Behavioral Cloning

## Project goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./output/model.png "Model Visualization"
[loss]: ./output/loss.png "Training/validation loss"
[camera-images]: ./output/camera-images.png "Sample camera images"

## Submitted files

The submission includes the following files: 

* model.py contains the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

Using the Udacity provided simulator and the `drive.py` script, the car can be
driven autonomously around the track by executing 
```sh 
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution
neural network. The file shows the pipeline I used for training and validating
the model, and it contains comments to explain how the code works.

## Model architecture 

I decided to use the model architecture described in
[this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf),
developed by NVidia engineers. In summary, the model contains 6 convolutional
layers, and 3 fully-connected ones. The model doesn't use max pooling, but
employs 2-strides for the first 3 convolutional layers.  For fully-connected
layers, the model uses ReLU to introduce nonlinearity. There are also 2
pre-processing layers (cropping and normalizing), desribed below in this
document.

![Model architecture][model]

## Model training

For training, I used Adagrad optimizer with the learning rate of 0.001, which
showed good convergence on the training dataset. 

To control overfitting, I split the dataset into the training and validation
subsets. The training process calculates the training and validation loss
changes during training: 

![Loss visualization][loss]

This plot shows that there are no signs of overfitting, due to the diversity in
training data, recorded on both track 1 and 2. Therefore, no additional effort
is required to address overfitting problems.

Concerning the number of epochs, it looks like 10 epochs is a good trade-off
between training time and model performance. I haven't noticed any perceived
improvements in car behaviour when the number of epochs was above 10, even
though the training and validation errors continued to decrease. 

## Training data collection strategy

The purpose of this project is to build a driving model that would generalize
well. To achieve this, I decided to deliberately avoid recording the driving
along a testing route, which is a track 1 in a counter-clockwise
direction. Instead, I recorded a few circles of driving along the track 2, and
also 1 round of driving by the track 1 in the reverse direction. 

My idea to avoid recording the testing route directly was that, if I manage to
train the model to drive the test route with the data from other routes, that
would be an indication that the model is general enough to drive along
prevoiusly unseen tracks. 

Later in the process I added a few pieces from the test track, where the model
was not performing well, apparently due to specific road conditions (dirt
borders). I also recorded a few recovery laps, to make the model learn how to
recover from going off-center. 

Here is the sample images from the left, center, and right cameras:

![Sample images][camera-images]

## Training data preprocessing

The training data undergoes the following transformations in order to achieve
good model performance: 

1. The training dataset is biased towards zero steering angle, so I remove a
portion of such samples. Experimentally I found that I should remove about 30%
of zero samples (see function `balance_samples()` in `model.py` for details). 

2. Images from side cameras were also employed during training. Experimentally I
found that correcting the steering angle by 0.2 was a good value to improve
model's stability. I should mention that the model performed reasonably well
without using the side camera images, but I noticed that using them would
make the car more stable, especially on the straight segments of the track.

3. The images are cropped, removing the upper part that does not contain
valuable information. I used Keras' `Cropping2D` layer for that. 

4. The pixel data is normalized to the [0, 1] range. I used `Lambda` layers to
do that.

## Test drive

The final model meets the quality criteria for this project, tested on both
track 1 and track 2. See the video recordings `track-1.mp4` and `track-2.mp4` in
the directory `output`.
