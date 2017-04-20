#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataAnalysis/exploratoryAnalysis.png "Visualization"
[image2]: ./dataAnalysis/dataAugumentation.jpg "Various Data Augumentations"
[image3]: ./dataAnalysis/traindataaugumented.png "Final data distribution"
[image4]: ./downloadedImages/SpeedLimit30.png "Traffic Sign 2"
[image5]: ./downloadedImages/SpeedLimit50.png "Traffic Sign 3"
[image6]: ./downloadedImages/SpeedLimit60.png "Traffic Sign 4"
[image7]: ./downloadedImages/SpeedLimit80.png "Traffic Sign 5"
[image8]: ./downloadedImages/stop.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the training/valid/test distribution of output labels.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

* As first step, I have identified and removed blank images from training data.
* As the initial training data was very less, i have generated additional data
* First, I generated additional data for output labels that are under represented so that the max difference between labels is not more than 1:3
* I have mostly used image shift and rotate to generate this data.
* Then I used additional techniques like zoom, brightness, contrast to increase the training data.
* All the above data augumentation techniques keep the original image content with little variations so that your model will become robust.
* The idea is to remove bias of over-represented labels while still maintaining initial distribution
* I converted every image to Grayscale because traffic signs depend on shape than color.
* As a last step, I normalized the image data because of numerical stability. If data is not normalized, then gradient descent will not happen smoothly causing the training process to be very slow or worse diverge.

Here is an example of a traffic sign image with various image transformations used.

![alt text][image2]

 
Here is the distribution of training labels after data augumentation

![alt text][image3]

The original data set had 34799 images and the augmented data set had 374004 images.


####2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 6x6     	| 1x1 stride, same padding, outputs 32x32x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x24 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x48 				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 	    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 			    	|
| Fully connected		| output channels 200        					|
| RELU					|												|
| Dropout				| pkeep 0.75									|
| Fully connected		| output channels 43        					|
| Softmax				| 	        									|

 


####3. Training the model

To train the model, I used an AdamOptimizer with batch size of 128 and 20 epochs. I used decaying learning rate ranging from 0.0001 to 0.005 with decay speed of 10000

####4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

I have started with Lenet architecture. since accuracy was low i went deep and added additional conv layer. To go further deep, i changed the depth of each conv layer to increasingly progressive to capture more image features. I also used max pooling after each conv layer to reduce the features instead of larger conv strides or valid padding as both of those are aggresive ways for feature reduction. I introduced dropout with pkeep of 75% in fully connected layer for regularization.I also increased the depth of output layer to account for increased output labels. I also used decaying learning rate which improved my acccuracy.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 97.21
* test set accuracy of 95.52


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because its not positioned center and is small towards one side of image. all other images must be easy to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed Limit 30    	| U-turn 										|
| Speed Limit 50		| Yield											|
| Speed Limit 60	    | Bumpy Road					 				|
| Speed Limit 80		| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|



