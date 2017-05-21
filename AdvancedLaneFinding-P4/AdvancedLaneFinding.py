import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import glob
from moviepy.editor import VideoFileClip
get_ipython().magic('matplotlib inline')


#Camera Calibration
#Takes in List of chessboard Images taken from your camera
#and calculates its calibration and returns camera matrix, distortion coefficients
#in below function we are looking at chess board images of 6 * 9
def calibrateCameraWithImages(Images):
    
    obj_points = []
    img_points = []

    objp = np.zeros((6 * 9, 3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) #x ,y

    for image in Images:
        img = pimg.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)
            cv2.drawChessboardCorners(img,(9,6),corners,ret)
    
    ret, mtx, dist, rvecs, tvecs =  cv2.calibrateCamera(obj_points,img_points,img.shape[0:2],None,None)
    return mtx, dist



#Returns an undistored Image given a camera image, camera matrix, distortion coefficients
def cal_undistort(img, mtx, dist):    
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist



#vizualize a sample camera image and its undistored image
def viz_camera_calib(mtx, dist):
    input_image = pimg.imread("data/camera_cal/calibration1.jpg")
    undistorted = cal_undistort(input_image, mtx, dist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(input_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



#Sobel Gradient Filtering
# Takes in arguments for orient(x-axis / y-axis), threshold values to filter, kernel size and color channel to work on
#you can choose to work on GRAY image or convert to HLS channel and choose S channel Image
def sobel_thres(src,orient = 'x',thres = (50,120),color_channel='GRAY',kernel=3):
    
    if color_channel == 'GRAY':
        gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    else:
        hls = cv2.cvtColor(src,cv2.COLOR_RGB2HLS)
        gray = hls[:,:,2]
    sobel = cv2.Sobel(gray,cv2.CV_64F,orient == 'x',orient == 'y',ksize=kernel)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(sobel_abs * 255 / np.max(sobel_abs))
    sobel_binary = np.zeros_like(sobel_scaled)
    sobel_binary[(sobel_scaled >= thres[0]) & (sobel_scaled <= thres[1])] = 1
    return sobel_binary



#Sobel Gradient filtering on combined x and y axis
# combines X and Y axis sobel gradients and then perfomrs threshold filtering
def sobel_thres_magnitude(src,thres = (50,120),color_channle='GRAY',kernel=3):   
    if color_channle == 'GRAY':
        gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    else:
        hls = cv2.cvtColor(src,cv2.COLOR_RGB2HLS)
        gray = hls[:,:,2]
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernel)
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    sobel_scaled = np.uint8(sobel_mag * 255 / np.max(sobel_mag))
    sobel_binary = np.zeros_like(sobel_scaled)
    sobel_binary[(sobel_scaled >= thres[0]) & (sobel_scaled <= thres[1])] = 1
    return sobel_binary
    



#Converts given image to HLS channel 
#chooses the channel passed and performs color filtering on threshold values passed
def color_spaces(src,color_space = 'S',thres=(150,255)):
    hls = cv2.cvtColor(src,cv2.COLOR_RGB2HLS)
    if color_space == 'H':
        i = 0
    elif color_space == 'L':
        i = 1
    else:
        i = 2
    sub_img = hls[:,:,i]
    color_binary = np.zeros_like(sub_img)
    color_binary[(sub_img > thres[0]) & (sub_img <= thres[1])] = 1
    return color_binary



#for a given image, it returns the image with only region of interest displayed
# for viszualization purpose
def region_of_interest(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    vertices = np.array([[(np.int(img_width * 0.10 ),np.int(img_height * 0.90)),
                    (np.int(img_width * 0.40 ),np.int(img_height * 0.65 )),
                    (np.int(img_width * 0.60 ),np.int(img_height * 0.65 )),
                    (np.int(img_width * 0.90 ),np.int(img_height * 0.90))]],dtype=np.int32)
    mask = np.zeros_like(image)       
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255        
    cv2.fillPoly(mask, vertices, ignore_mask_color)   
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



# for a given image, it identifies the region of interest and transforms its perspective to birds-eye view
# this will avoid front view distortion issues of lane lines
# the function also returns Minv to get back the original image if needed later
def perspective_transoform(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    img_size = (image.shape[1], image.shape[0])
    #print(image.shape)
    offset = np.int(img_height * 0.075)
    src = np.float32([[np.int(img_width * 0.10 ),np.int(img_height * 0.90)],
                    [np.int(img_width * 0.40 ),np.int(img_height * 0.65 )],
                    [np.int(img_width * 0.60 ),np.int(img_height * 0.65 )],
                    [ np.int(img_width * 0.90 ),np.int(img_height * 0.90)]])
    dst = np.float32([ [offset, img_size[1]-offset],[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size)
    return warped,Minv
    



# this function defines the pipeline used to binary image
# For a given image, this function performs various color and gradient thresholdings
#combines them in meaningful way and returns final Binary image used to identify lane lines
#Data Pipeline
    #calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on Gray Image
    #calculate Sobel Gradient for X-axis, Y-axis, X and Y Magnitude on S channel Image of HLS
    #get color thresholding filtered image of S channel
    #Img1 : Gray Scale X-axis and Y-axis combined
    #Img2 : S channel X-axis and Y-axis combined
    #Img3 : Color Channel and Gray Magnitude combined
    #Img4 : Color Channel and S channel Magnitude combined  
    #Final Binary Image: OR combination of (Img1,Img2,Img3,Img4) 
    
def all_combinations(src,grad_thres=(50,120),color_thres=(50,255),kernel=9):
    sobelx_gray = sobel_thres(src,'x',grad_thres,'GRAY',kernel)
    sobely_gray = sobel_thres(src,'y',grad_thres,'GRAY',kernel)
    sobel_mag_gray = sobel_thres_magnitude(src,grad_thres,'GRAY',kernel)
    sobelx_S = sobel_thres(src,'x',grad_thres,'S',kernel)
    sobely_S = sobel_thres(src,'y',grad_thres,'S',kernel)
    sobel_mag_S = sobel_thres_magnitude(src,grad_thres,'S',kernel)
    color_channel = color_spaces(src,'S',color_thres)
    combined_binary = np.zeros_like(sobelx_gray)
    combined_binary[((sobelx_gray == 1) & (sobely_gray == 1)) | 
                    ((sobelx_S == 1) & (sobely_S == 1)) | 
                    ((sobel_mag_S == 1) & (color_channel == 1)) |
                    ((sobel_mag_gray == 1) & (color_channel == 1))] = 1
    return combined_binary
    



#vizualize each step of pipeline
#Optional
def visualize_pipeline(src,grad_thres=(50,120),color_thres=(50,255),kernel=9):
    

    sobelx_gray = sobel_thres(src,'x',grad_thres,'GRAY',kernel)
    sobely_gray = sobel_thres(src,'y',grad_thres,'GRAY',kernel)
    sobel_mag_gray = sobel_thres_magnitude(src,grad_thres,'GRAY',kernel)
    sobelx_S = sobel_thres(src,'x',grad_thres,'S',kernel)
    sobely_S = sobel_thres(src,'y',grad_thres,'S',kernel)
    sobel_mag_S = sobel_thres_magnitude(src,grad_thres,'S',kernel)
    color_channel = color_spaces(src,'S',color_thres)
    combined_binary = all_combinations(src,grad_thres,color_thres,kernel)
    
    pic_region = region_of_interest(src)
    pic_warped, p_Minv = perspective_transoform(src)
    binary_warped, Minv = perspective_transoform(combined_binary)
    
    # Plot the result
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9),(ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(25, 25),sharex='col', sharey='row')
    f.tight_layout()
    ax1.imshow(src)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(sobelx_gray, cmap='gray')
    ax2.set_title('X axis: Thresholded Gradient Gray Channel', fontsize=15)
    ax3.imshow(sobely_gray, cmap='gray')
    ax3.set_title('Y axis: Thresholded Gradient Gray Channel', fontsize=15)
    ax4.imshow(sobel_mag_gray, cmap='gray')
    ax4.set_title('combined: Thresholded Gradient Gray Channel', fontsize=15)



    ax5.imshow(sobelx_S, cmap='gray')
    ax5.set_title('X axis: Thresholded Gradient S Channel', fontsize=15)
    ax6.imshow(sobely_S, cmap='gray')
    ax6.set_title('Y axis: Thresholded Gradient S Channel', fontsize=15)
    ax7.imshow(sobel_mag_S, cmap='gray')
    ax7.set_title('combined: Thresholded Gradient S Channel', fontsize=15)

    ax8.imshow(color_channel, cmap='gray')
    ax8.set_title('Color Channel S', fontsize=15)
    ax9.imshow(combined_binary, cmap='gray')
    ax9.set_title('Combined', fontsize=15)
    
    ax10.imshow(pic_region)
    ax10.set_title('Region of Interest', fontsize=15)
    ax11.imshow(pic_warped)
    ax11.set_title('Perspective Transform: ROI', fontsize=15)
    ax12.imshow(binary_warped, cmap='gray')
    ax12.set_title('Perspective Transform: Binary Image', fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



#For given Binary image,Returns X axis and y axis Pixel positions of left line and right line
# uses Sliding window approach
def get_pixelpositions_withSlidigWindow(binary_warped):
    
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])   # base position of left lane line
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # base position of right lane line
    
    #assigning values to x and y meter convertion global variables
    # need assignment only in first run and values will be constant throught the video processing
    global ym_per_pix
    global xm_per_pix
    
    ym_per_pix = 30/binary_warped.shape[0] if ym_per_pix is None else ym_per_pix # meters per pixel in y dimension
    xm_per_pix = 3.7/(rightx_base -leftx_base) if xm_per_pix is None else xm_per_pix # meters per pixel in x dimension 

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = np.int(binary_warped.shape[1] * 0.075)  # margin to check
    minpix =  np.int(binary_warped.shape[0] * 0.05) # min points
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    return leftx,lefty,rightx,righty
    



#given image, polymonial fit coefficients of previous left and right lane
#identifies the current image x and y values of left and right lane
#Assumption  is that, Lane Lines will not change its position significantly from last frame
def get_pixelpositions_withPreviousLinefits(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = np.int(binary_warped.shape[1] * 0.075) # margin with in which to check for lane lines instead of whole image
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx,lefty,rightx,righty



#Visualization of sliding window
def viz_sliding_windows(nonzeroy,nonzerox,left_lane_inds,right_lane_inds):
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)



#Curvature Calculation
#Returns strings to be printed on output Image
#curvature radious and position offset

def get_curve_offset(binary_warped,lefty,leftx,righty,rightx):

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty) # bottom of Image

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    
    #car offset from center
    # calculate Lane center, Image center in meters and do subraction
    ybottom_in_meters = y_eval*ym_per_pix
    left_bottomx = left_fit_cr[0]*ybottom_in_meters**2 + left_fit_cr[1]*ybottom_in_meters + left_fit_cr[2]
    right_bottomx = right_fit_cr[0]*ybottom_in_meters**2 + right_fit_cr[1]*ybottom_in_meters + right_fit_cr[2]
    lane_center = left_bottomx + (right_bottomx - left_bottomx) / 2
    image_center = binary_warped.shape[1] * xm_per_pix / 2
    offset = image_center - lane_center

    #print(offset)
    curve = "Radius of Curvature = {:0.2f} (m)".format(min(left_curverad,right_curverad))
    direction = 'left of center' if offset < 0.0 else 'right of center' if offset > 0.0 else 'in the center'
    position = 'Vehicle is {:0.2f}m '.format(np.absolute(offset)) +direction
    return curve,position



#Draws the final output image stitched in video
#Takes input image, binary warped image, left and right lane x-axis positions
#Minv for calculating back original image from warped image
#curve and position strings to be printed on output image
def draw_output_image(inputImage,binary_warped,left_fitx,right_fitx,Minv,curve,position):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(inputImage, 1, newwarp, 0.3, 0)
 
    #put text string on output image
    cv2.putText(result,curve,(100,50),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    cv2.putText(result,position,(100,100),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    #plt.imshow(result)
    return result



#Function that takes input image and return output image
# contains entire pipeline of processing
# 1. get undistorted image of input
# 2. get final binary image with all color and gradient filtering
# 3. get binary warped image using perspective transform
# 4. get x and y axis pixel positions of left and right lanes from binary warped image
# 5. calculate the polynomial fit coefficients of left and right lanes
# 6. get the exact x and y axis pixel positions of left and right lane
# 7. get the radius curvature and vehicle offset from center
# 8. get the final output image

def get_output_image(input_image):
    src = cal_undistort(input_image, mtx, dist)
    processed_binary_image = all_combinations(src,grad_thres,color_thres,kernel)
    binary_warped, Minv = perspective_transoform(processed_binary_image)
    
    global previous_left_fit
    global previous_right_fit
    
    if len(previous_left_fit) > 0 and  len(previous_right_fit) > 0:
        leftx,lefty,rightx,righty = get_pixelpositions_withPreviousLinefits(binary_warped,previous_left_fit,previous_right_fit)
    else:   
        leftx,lefty,rightx,righty = get_pixelpositions_withSlidigWindow(binary_warped)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    previous_left_fit = left_fit     # use left fit in next frame
    previous_right_fit = right_fit   # use right fit in next frame 

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    curve,position = get_curve_offset(binary_warped,lefty,leftx,righty,rightx)
    output_image = draw_output_image(src,binary_warped,left_fitx,right_fitx,Minv,curve,position)
    return output_image




#Main method starts here

# Camera calibration step
Images = glob.glob('data/camera_cal/calibration*.jpg')
mtx, dist = calibrateCameraWithImages(Images)   

#variables
grad_thres=(50,120)   # Thresholds for Gradient Filtering
color_thres=(120,255)# Thresholds for Color Filtering
kernel=5

ym_per_pix = None   #Y-axis image to meter conversion value
xm_per_pix = None   #X-axis image to meter conversion value

previous_left_fit = []   # polynomial fit coefficients of previous left lane
previous_right_fit = []  # polynomial fit coefficients of previous Right lane


white_output = 'data/test_videos_output/project_video.mp4'
clip1 = VideoFileClip("data/test_videos/project_video.mp4")
white_clip = clip1.fl_image(get_output_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')

