#**Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/combinedImages.jpg "Data Augumentation"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

my model (lines 136-166) consists of a 
* Cropping layer -- Top 60 and bottom 20 rows of image are removed
* Normalization layer -- (x/255.0 - 0.5)
* 4 convolutional layers each with relu activation followed by max pooling
* 3 fully connected layers with relu activations. dropout used after first two fc layers to reduce overfitting
* 1 output channel which predicts the car angle

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 68-72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 169).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


The overall strategy for deriving a model architecture was to try different architectures and find one that best fits the problem.

My first step was to use a convolution neural network model similar to the one i used for Traffic sign classification project and see how it performs on different problem set

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropouts in fully connected layers.

I used generators to train my model. this was helpful because i didn't have to resize my image and didn't face memory issues. 

The final step was to run the simulator to see how well the car was driving around track one. For many attempts with slightly different model architectures the vehicle fell off the track at a particular curve.

Later I used Data Augumentation technique to duplicate image entries where steering angle was high. This gave model more data to learn how to drive at curves
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py llines 136-166) consisted of a convolution neural network with 4 conv and 3 fc layers and 1 output channel.

Here is a visualization of the architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							|
| Cropping         		| 80x320x3 image   								| 
| Lambda         		| Normalized image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 80x320x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 40x160x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 40x160x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 20x80x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 20x80x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x40x128 			    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x40x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x20x128 			    |
| Fully connected		| output channels 300        					|
| RELU					|												|
| Dropout				| pkeep 0.75									|
| Fully connected		| output channels 100        					|
| RELU					|												|
| Dropout				| pkeep 0.75									|
| Fully connected		| output channels 10        					|
| RELU					|												|
| Fully connected		| output channels 1        						|


####3. Creation of the Training Set & Training Process

I collected Training data in two parts.
* first i drove in middle of lane for 2 laps to record good driving behaviour
* next i drove only at curves to get more data of driving at curves as i felt its very important to train a successful model

Then I used two data augumentation methods to create additional training data.
* duplicate data for car angles that are not zero. higher absolute values of angle occur very rare so used higher duplication for those.
* use left and right camera images as well as flip images.

below is the image with data augumentation


![alt text][image1]


My actual data points collected from driving was 6477. After above Data augumentations, I had 79860 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by running model at incremental epochs and finding the best solution at 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
