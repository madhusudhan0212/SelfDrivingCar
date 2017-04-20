#**Traffic Sign Recognition** 

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
[image2]: ./dataAnalysis/dataAugumentation1.jpg "Various Data Augumentations"
[image3]: ./dataAnalysis/traindataaugumented.png "Final data distribution"
[image4]: ./downloadedImages/SpeedLimit30.jpg "Traffic Sign 2"
[image5]: ./downloadedImages/speedLimit50.jpeg "Traffic Sign 3"
[image6]: ./downloadedImages/SpeedLimit60.jpeg "Traffic Sign 4"
[image7]: ./downloadedImages/speedLimit80.jpeg "Traffic Sign 5"
[image8]: ./downloadedImages/stop.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the training/valid/test set distribution of output labels. The output labels are on x-axis and their counts are on y-axis

![alt text][image1]

###Design and Test a Model Architecture

####1. Image data preprocessing. 

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

The original data set had 34799 images and the augmented data set had 373848 images.


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

I have started with Lenet architecture. since accuracy was low i went deep and added additional conv layer. To go further deep, i changed the depth of each conv layer to increasingly progressive to capture more image features. I also used max pooling after each conv layer to reduce the features instead of larger conv strides or valid padding as both of those methods are aggresive ways for feature reduction. I introduced dropout with pkeep of 75% in fully connected layer for regularization. I increased the depth of output layer to account for increased output labels and used decaying learning rate which improved my acccuracy.

My final model results were:
* training set accuracy of 99.99 %
* validation set accuracy of 97.78 %
* test set accuracy of 95.91 %


###Test a Model on New Images

####1. I have downloaded five German traffic signs found on the web and tried to classify below. For each image, I discuss what was model's prediction.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because its not positioned center and is small towards one side of image. all other images must be easy to classify.

####2. Model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| Speed limit (30km/h)  | Speed limit (30km/h) 							|
| Speed limit (50km/h)	| Roundabout mandatory							|
| Speed limit (60km/h)	| Speed limit (60km/h)			 				|
| Speed limit (80km/h)	| Speed limit (80km/h) 							|
| Stop      			| Stop   										|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.9%

####3. Model Predictions analysis

The code for making predictions on my final model is located in the 45th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Speed limit (30km/h) (probability of 0.999), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Speed limit (30km/h)   						| 
| .00     				| Speed limit (80km/h) 							|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (50km/h)					 		|
| .00				    | Speed limit (100km/h)      					|


For the second image the model predicted the sign as Roundabout Mandatory instead of Speed limit (50km/h) sign. As mentioned earlier, this image was difficult to predict as sign symbol was very small and towards one corner of image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Roundabout mandatory   						| 
| .04     				| Priority road 								|
| .00					| Keep right									|
| .00	      			| Speed limit (60km/h)			 				|
| .00				    | No entry      								|

For the third image, the model is sure that this is a Speed limit (60km/h) (probability of 0.92), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .92         			| Speed limit (60km/h)   						| 
| .06     				| Speed limit (50km/h) 							|
| .006					| Bicycles crossing								|
| .004	      			| Turn left ahead					 			|
| .001				    | Children crossing      						|

For the fourth image, the model is very sure that this is a Speed limit (80km/h) (probability of 0.988), and the image does contain a Speed limit (80km/h) sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .988         			| Speed limit (80km/h)   						| 
| .006     				| Speed limit (100km/h) 						|
| .005					| Speed limit (60km/h)							|
| .00	      			| Speed limit (120km/h)					 		|
| .00				    | Speed limit (50km/h)      					|

For the fifth image, the model is very sure that this is a Stop sign (probability of 0.999), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop   								   		| 
| .00     				| No entry 										|
| .00					| No passing									|
| .00	      			| Speed limit (70km/h)					 		|
| .00				    | Speed limit (50km/h)      					|



