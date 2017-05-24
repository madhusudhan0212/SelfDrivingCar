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
[image8]: ./data/pipeline_images/whiteYellow.jpg "Yellow and White Images"
[image4]: ./data/pipeline_images/binaryOutput.jpg "Binary Output"
[image5]: ./data/pipeline_images/perspective_transform.jpg "Perspective Transform"
[image6]: ./data/pipeline_images/polynomialFits.jpg "Polynomial Fits"
[image7]: ./data/pipeline_images/finalOutput.jpg "Output Image"
[video1]: ./data/output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is contained in lines 14 through 32 of the file called `AdvancedLaneFinding.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline 

#### 1. example of a test image and its distortion-corrected image.

![alt text][image2]

#### 2. Data Pipeline

I used a combination of color and gradient thresholds to generate a binary image (function `all_combinations()`, which appears in lines 253 through 282 in the file `AdvancedLaneFinding.py`).  Below steps explain various steps of pipeline

    * calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on Gray Image
    * calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on S channel Image of HLS
    * get color thresholding filtered image of S channel
    * Img1 : Gray Scale X-axis and Y-axis combined
    * Img2 : S channel X-axis and Y-axis combined
    * Img3 : Color Channel and Gray Magnitude combined
    * Img4 : Color Channel and S channel Magnitude combined  
    * Binary Image I: OR combination of (Img1,Img2,Img3,Img4)
    * Img5 : White color in Region of Interest
    * Img6 : V channel of YUV that identify Yellow color in region of interest
    * Img7 : OR combination of Img5 and Img6
    * Final Binary Image: OR combination of (Binary Image I,Img7) 

Below Image shows various pipeline images of Input image
![alt text][image3]

Below Image shows images of White and Yellow lines
![alt text][image8]

The FInal Binary image will look like this
![alt text][image4]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `perspective_transoform()`, which appears in lines 169 through 184 in the file `AdvancedLaneFinding.py`.  The `perspective_transoform()` function takes as inputs an image and gives warped output.   I chose to hardcode the source and destination points in the following manner:

```python
img_height = image.shape[0]
img_width = image.shape[1]
img_size = (image.shape[1], image.shape[0])

offset = np.int(img_height * 0.075)
src = np.float32([[np.int(img_width * 0.10 ),np.int(img_height * 0.97)],
                    [np.int(img_width * 0.40 ),np.int(img_height * 0.65 )],
                    [np.int(img_width * 0.60 ),np.int(img_height * 0.65 )],
                    [ np.int(img_width * 0.90 ),np.int(img_height * 0.97)]])
dst = np.float32([[offset, img_size[1]-offset],
    			[offset, offset], 
    			[img_size[0]-offset, offset],
                [img_size[0]-offset, img_size[1]-offset]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 128, 698      | 54, 666       | 
| 512, 468      | 54, 54      	|
| 768, 468     	| 1226, 54      |
| 1152, 698     | 1226, 666     |

I verified that my perspective transform was working as expected by drawing a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Identify lane-line pixels

I did this in functions `get_pixelpositions_withSlidigWindow()` and `get_pixelpositions_withPreviousLinefits()` in lines 340 through 427 in my code in `AdvancedLaneFinding.py`. The first method used sliding windows approach to identify lane lines. The second method takes in previous frame's predicted lines and narrows its search area to find lane lines in new frame. Both functions return identify and return current image's x and y pixel positions of left and right lane.
The pixel points are then used to fit lane lines with 2nd order polynomial. Below image shows the sliding window approach used and lane lines pixels identified and fit lines

Then both left and right lines are then subjected to sanity checks. 
* The previous predicted line and current predicted line are with in close margin
* The predicted line is not in unexpected areas

If the sanity check fails then predictions from previous frames are used.
If sanity check fails continuosly for more than 7 frames then lanes are identified from scrath using sliding window.

![alt text][image6]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `get_curve_offset()` in lines 447 through 474 in my code in `AdvancedLaneFinding.py`. To identify vehicl position with respect to center, I calculated lane center and image center in meters and did subraction. For radius curvature, i identified the curvature for both left and right lanes and used minimum of both to display on output image.

#### 6. final output

I implemented this step in function `draw_output_image()` in lines 481 through 505 in my code in `AdvancedLaneFinding.py`.  In this step, i warped the detected lane boundaries back onto the original image and added position and curvature strings back onto original image

Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Below is the link to my output video. It works reasonably well in identifying lane lines

Here's a [link to my video result](./data/output_videos/project_video_attempt2.mp4)

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* Initially my pipeline found it difficult to identify lane lines under bright light. But the issue got fixed when I included Gradient Filtering on S channel into my pipeline
* you can notice it in pipeline images. The gradient threshold images of Gray scale image result almost blank images and lane lines were identified from S channel images
* The pipeline has major dependency on region of interest to identify lanes. The pipeline will fail if region of interest doesnt contain lane line. ( sharp turns, during lane changes, not so clean lane lines)
* Few improvements to make pipeline more robust is to remember several previous predictions and use an average to have smooth predictions.
