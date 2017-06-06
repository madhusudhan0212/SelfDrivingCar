##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Prepare labeled data from vehicle and non-vehicle image data sets
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Apply image resize and ravel and append color features, as well as histograms of color, to the HOG feature vector.
* Normalize the features and randomize a selection for training and testing.
* Train a Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./data/pipeline_Images/Vehicle.png
[image2]: ./data/pipeline_Images/nonVehicle.png
[image3]: ./data/test_images/test4.jpg
[image4]: ./data/hog_test_images/4.png
[image5]: ./data/pipeline_Images/4.png
[image6]: ./data/test_images_output/test4.jpg
[video1]: ./data/output_videos/Projectvideo.mp4

---

###Prepare Labeled data

The first step of this project is to prepare labeled data from vehicle and non-vehicle data sets provided. I have read all vehicle images into cars list and non vehicle images into notcars list. The code for this step is contained in the 8 and 9 code cells of the IPython notebook 

###Histogram of Oriented Gradients (HOG)

####1. HOG features extraction from the training images.

The code for this step is contained in the 7 and 11 code cells of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle
![alt text][image1]

Not Vehicle
![alt text][image2]

* I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  After some iterations with different values for color space, orientation, pixels_per_cell and cells_per_block, i found that i was getting best test accuracy of model consistently when i used below values for the parameters.

`YCrCb` color space , `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


* I have also resized the image to `(32,32)` and ravel the image to get all pixel values as features (Ipython cell 2)
* I have also used color transform to `YCrCb` and extracted histograms of `32 bins` for each color channel (Ipython cell 3)
* I then combined all three ( hog features, pixel value features and color histogram features) to get the final features (total feature vecotr length: 8556)
* Now i normalized the features using StandardScaler and randomly selected 20% of data aside for testing the model (Ipython cell 12 and 13)
* I trained a linear SVM using lower C value (0.1) for higher margin Hyper plane. I chose linear SVM because of its speed compared to SVM with linear kernel. I got a test accuracy of 98.94 % (Ipython cell 14 and 15)

---
### Detecting vehicles in video stream
---

Here i will show how to identify vehicles in new images. lets find how our model works on below test image.

![alt text][image3]


###Sliding Window Search

The method i used to identify vehicles quickly in new images is Sliding Window Search. In this method, you select the region of interest of your image (part of image where you expect to find vehicles, usually the bottom half of the image) and perform Hog feature extraction only once for each color channel with parameter feature_vec=False. Now you use sliding window approach to check each window for vehicle detection by extracting hog features, pixel features and color histogram features for that particular window. The major advantage of Sliding Window Search is that you calculate Hog features only once for each channel.

I used 3 different scales `[0.9,1.5,1.9]` with `2` cells_per_step (defines the cells to step during sliding window search). These scale values are used to appropriately scale the input image and then perform sliding window search on the scaled image. I have arrived at these scale values after using different scale values on test images and observing the output.

The code for this step is in Ipython cells 6 and 16.

Below image shows the HOG features in the region of interest for each channel of the input image

![alt text][image4]

after sliding window search, you will get list of positions of positive detections where vehicles are expected for each scale. I then created a heatmap from all the positive detections. Below image shows positive detections for each scale and heatmap image

![alt text][image5]

From above image, red boxes are from 0.9 scale, green boxes are from 1.5 scale and blue boxes are from 1.9 scale. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Once the bounding box is constructed, to avoid any false positives, i check the width and height of each final bounding box constructed from the heatmap and check if the width and height values are in expected range. (Ipython cell 18)

Below image shows the final output image with vehicles identified. Note that false positive is not present in final output

![alt text][image6]


The Ipython cell 10 shows the list of all the parameters used for the pipeline. Additional test images, corresponding hog images, pipeline images and output images can be found in data folder of the repository


### Video Implementation

Each frame of the video is an image and i performed all the above steps to each of the image. Below are the few additional optimization techniques i used on video stream.

* I stored the bounding boxes of last 30 frames and each time i used all of them to generate heatmap
* I then used a threshold of 18 to filter out any false positives
* As an additional layer to filter out any false positives, i check the width and height of each final bounding box constructed from the heatmap and check if the width and height values are in expected range.


With above optimizations, i was able to find vehicles and draw smooth bounding boxes around them in the project video reasonably well. Below is the link to the final output video.
[link to my video result](./data/output_videos/Projectvideo.mp4)


---

###Discussion

Initially my pipeline was not detecting white car very well. so i generated a bunch of positive labels with white car from project video and included it in training data. after that model's white car detection improved significantly.The pipeline will not work well in heavy traffic where there will be lot of cars with overlap and you see only part of each car. If i want to pursue this project further, i would gather more training data in heavy traffic setting and use deep learning models to improve accuracy.

