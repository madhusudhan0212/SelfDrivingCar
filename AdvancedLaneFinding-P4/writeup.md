## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./data/pipeline_images/cameraCalibration.jpg "Camera Calibration"
[image2]: ./data/pipeline_images/undist_Input.jpg "Input Image"
[image3]: ./data/pipeline_images/pipeline.jpg "Pipeline Images"
[image4]: ./data/pipeline_images/binaryOutput.jpg "Binary Output"
[image5]: ./data/pipeline_images/perspective_transform.jpg "Perspective Transform"
[image6]: ./data/pipeline_images/polynomialFits.jpg "Polynomial Fits"
[image7]: ./data/pipeline_images/finalOutput.jpg "Output Image"
[video1]: ./data/output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is contained in lines 14 through 32 of the file called `AdvancedLaneFinding.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline 

#### 1. example of a distortion-corrected image.

![alt text][image2]

#### 2. Data Pipeline

I used a combination of color and gradient thresholds to generate a binary image (function `all_combinations()`, which appears in lines 170 through 183 in the file `AdvancedLaneFinding.py`).  Below steps explain various steps of pipeline

    * calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on Gray Image
    * calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on S channel Image of HLS
    * get color thresholding filtered image of S channel
    * Img1 : Gray Scale X-axis and Y-axis combined
    * Img2 : S channel X-axis and Y-axis combined
    * Img3 : Color Channel and Gray Magnitude combined
    * Img4 : Color Channel and S channel Magnitude combined  
    * Final Binary Image: OR combination of (Img1,Img2,Img3,Img4) 

Below Image shows various pipeline images of Input image
![alt text][image3]

The final Binary image will look like this
![alt text][image4]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `perspective_transoform()`, which appears in lines 127 through 152 in the file `AdvancedLaneFinding.py`.  The `perspective_transoform()` function takes as inputs an image and gives warped output.   I chose the hardcode the source and destination points in the following manner:

```python
	img_height = image.shape[0]
 	img_width = image.shape[1]
    img_size = (image.shape[1], image.shape[0])

    offset = np.int(img_height * 0.075)
    src = np.float32([[np.int(img_width * 0.10 ),np.int(img_height * 0.90)],
                    [np.int(img_width * 0.40 ),np.int(img_height * 0.65 )],
                    [np.int(img_width * 0.60 ),np.int(img_height * 0.65 )],
                    [ np.int(img_width * 0.90 ),np.int(img_height * 0.90)]])
    dst = np.float32([ [offset, img_size[1]-offset],
    					[offset, offset], 
    					[img_size[0]-offset, offset],
                        [img_size[0]-offset, img_size[1]-offset]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 128, 648      | 54, 666       | 
| 512, 468      | 54, 54      	|
| 768, 468     	| 1226, 54      |
| 1152, 648     | 1226, 666     |

I verified that my perspective transform was working as expected by drawing a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Identify lane-line pixels

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `get_curve_offset()` in lines 348 through 375 in my code in `AdvancedLaneFinding.py`

#### 6. final output

I implemented this step in function `draw_output_image()` in lines 383 through 407 in my code in `AdvancedLaneFinding.py`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Below is the link to my output video. It works reasonably well in identifying lane lines

Here's a [link to my video result](./data/output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
