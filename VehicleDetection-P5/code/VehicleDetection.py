import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
get_ipython().magic('matplotlib inline')




# function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Changes pixel values range if reading .png files with mpimg!
def check_and_scale(image):
    max_val = max(image.ravel())
    if max_val <= 1.0:
        image = (image * 255).astype(np.float32)
    return image


# takes an image and converts it to different color space
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# function to get Hog features of an Image
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        image = image[:,:,0:3]  # takes only color channels of depth 3.
        #check image and scale to (0-255) if in (0-1) scale
        image = check_and_scale(image)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features



# function to take folder path and return all the images in its subfolders
# constructs an list of all images used for training the model
def get_image_paths(path):
    images_list=[]
    for folder in os.listdir(path):
        folder_path = os.path.join(path,folder)
        if os.path.isdir(folder_path):
            for image in os.listdir(folder_path):
                image_name, ext = os.path.splitext(image)
                if ext in ('.png','.jpeg','.jpg'):
                    images_list.append(os.path.join(folder_path,image))
    return images_list



# Read in cars and notcars data set  
# Since the actual vehicles data is huge, i am reading it from outside Repository folder from local
Vehicles_data_folder = "../../../VehicleDetectionData/vehicles"
Non_Vehicles_data_folder="../../../VehicleDetectionData/non-vehicles"

cars = get_image_paths(Vehicles_data_folder)
notcars = get_image_paths(Non_Vehicles_data_folder)



### parameters for the pipeline.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

min_pixel_width = 48  # minimum box width in final output image
min_pixel_height = 48 # minimum box height in final output image

smoothing_over_frames = 30 # consider last n frames to draw boxes using heatmap
threshold = 18  # threshold value to show the box in output image
total_boxes_list = [None] * smoothing_over_frames # empty list to track bounding boxes over multiple frames



# extract features of Car data and Not car data
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)



#Normalize features
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))



# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print('size of test samples: ',len(y_test))



# Train the Model

svc = LinearSVC(C=0.1)   # using lower C value for higher margin Hyper plane

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')



# Calculate Accuracy score of your model
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


#                              ### FINDING VEHICLES IN VIDEO STREAM ###


# single function that can extract features using hog sub-sampling and make predictions
def find_cars(img1, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    bbox_list = []
    
    #check image and scale to (0-255) if in (0-1) scale
    img1 = check_and_scale(img1)
        
    #draw_img = np.copy(img1)
    img = img1.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    windows = [64]
    
    for window in windows:
        
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    
    
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))       
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:                   
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)          
                    bbox_list.append(((xbox_left, ytop_draw+ystart) ,(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return bbox_list



def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



### Finding Heatmaps and drawing one final box

def add_heat(heatmap, boxes_list):
    # Iterate through list of bboxes
    for bbox_list in boxes_list:
        if bbox_list is not None:
            for box in bbox_list:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Check if box has minimum height and width requirements
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzeroy) - np.min(nonzeroy)
        if width >= min_pixel_width and height >= min_pixel_height:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)     
    # Return the image
    return img



## Vizualize Pipeling images

def vizualize(out_img,heatmap,draw_img):
    fig = plt.figure(figsize=(15,15))
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(121)
    plt.imshow(out_img)
    plt.title('Find Cars Output')
    fig.tight_layout()
    global out_images_count
    fig.savefig(str(out_images_count) + '.png')
    out_images_count = out_images_count + 1



def process_input_image(image):
    ystart = np.int(image.shape[0] * .55)    #400     
    ystop = np.int(image.shape[0] * .95)     #656
    scale = [0.9,1.5,1.9]
    
    global total_boxes_list
    
    image_copy = np.copy(image)
    out_img_copy = np.copy(image)
    out_img = np.copy(image)
    
    box_list = []
    for i in range(len(scale)):
        box_list1 = find_cars(image_copy, ystart, ystop, scale[i], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list.append(box_list1)       

    
    #out_img = draw_boxes(out_img, box_list[0], color=(255, 0, 0), thick=4)
    #out_img = draw_boxes(out_img, box_list[1], color=(0, 255, 0), thick=5)
    #out_img = draw_boxes(out_img, box_list[2], color=(0, 0, 255), thick=6)
    

    box_list_comb = []
    for i in range(len(box_list)):
        box_list_comb = box_list_comb + box_list[i]
    
    
    total_boxes_list.pop(smoothing_over_frames-1)
    total_boxes_list.insert(0,box_list_comb)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,total_boxes_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(out_img_copy), labels)

    #vizualize(out_img,heatmap,draw_img)
    return draw_img
    



out_images_count = 1
test_input_folder = "../data/test_images"
test_output_folder="../data/test_images_output"

def viz_test_images_output(test_input_folder,test_output_folder,out_images_count):    
    for input_image in os.listdir(test_input_folder):
        image_name, ext = os.path.splitext(input_image)
        if ext in ('.png','.jpeg','.jpg'):
            image_path = os.path.join(test_input_folder,input_image)
            image = mpimg.imread(image_path)
            output_image = process_input_image(image)
            fig = plt.figure(figsize=(15,15))
            plt.imshow(output_image)
            fig.savefig(os.path.join(test_output_folder,input_image))
        
        
#viz_test_images_output(test_input_folder,test_output_folder,out_images_count)
    


output_video = '../data/output_videos/Projectvideo.mp4'
clip1 = VideoFileClip("../data/test_videos/project_video.mp4")
white_clip = clip1.fl_image(process_input_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(output_video, audio=False)')

