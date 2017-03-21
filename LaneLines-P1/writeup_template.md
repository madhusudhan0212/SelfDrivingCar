#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./examples/Pipeline_Explanation.jpg "Pipeline Images"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
* First, I converted the images to grayscale 
* applied Guassian smoothing with kernel size 5 
* used Canny Edge Detection with min and max thresholds of 80 and 130 
* identified region of interest (part of image where you expect to find Lane Lines) 
* finally used Hough Transform Line Detection to identify Lanes and draw them on top of input image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by following
* Categorized all input lines into positive slope lines and negative slope lines
* Identified and removed outlier slopes from both positive and negative slope lines
* Used numpy polyfit to identify one line each for all positive slope points and negative slope points
* Identify closest (close to camera) points for both positive and negative slope lines
* Identify farthest points for each side and if its less than a given fixed distance then extrapolate
* finally with closest and farther points for each side draw exactly two lines on to image identifying lanes

In below image, you can see the modifications done at each step of pipeline

![alt text][image2]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming with this approach is it fails
* to identify lane lines in snow, rain, sun reflections. 
* have exact vertical lanes in picture 
* entering a curve (at this point lanes will not be straight line but an arc) 

Another shortcoming could be our region of interest in Image. Lane Lines may not always be present in our region of interest.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to skip processing every single image in video and use info calculated from previous image since Lane lines will not have sharp edges. This will help in cases when we suddenly cannot identify lane lines due to sun reflection/mud on road, dashed lines are too far apart etc.

Another potential improvement could be to follow vehicles in front at safe distance when view is not clear
