import os
import csv

DataFoldername='Behaviour-Cloing-P-Data'
print('Code Directory: ',os.getcwd())

uplevelDir = os.path.dirname(os.getcwd())
print('Upper Level Directory: ', uplevelDir)

dataDirectory = os.path.join(uplevelDir,DataFoldername)
print('Data Directory: ', dataDirectory)


# In[5]:

#Data Augumentation I
#Since most of time car angle will be 0, our data set will be very biased to 0
# all other times the angle will be |angle| < 0.15 trying to keep car in lane
# there will be very few entries with |angles| > 0.3 (sharp turns) 
# since learnign small turns and sharp turns is very important for model i am  duplicating those entries

def getImagesMetaData(dataDirectory):
    dataDirectory_contents = os.listdir(dataDirectory)
    image_metadata = []
    angle_distribution = []
    actualLineCount = 0
    for filename in dataDirectory_contents:
        if filename.endswith('.csv'):
            csv_file_name = os.path.join(dataDirectory,filename)
            print('Reading CSV file: ',csv_file_name)
            with open(csv_file_name,'r') as csvfile:
                data = csv.reader(csvfile)
                for line in data:
                    angle = float(line[3])
                    if angle >= 0.15 or angle <= -0.15:
                        for i in range(2):
                            image_metadata.append(line)
                            angle_distribution.append(angle)
                    if angle >= 0.3 or angle <= -0.3:
                        for i in range(9):
                            image_metadata.append(line)
                            angle_distribution.append(angle)
                    image_metadata.append(line)
                    angle_distribution.append(angle)
                    actualLineCount = actualLineCount + 1
    return image_metadata,angle_distribution,actualLineCount
            

image_metadata,angle_distribution,actualLineCount = getImagesMetaData(dataDirectory)


print('Actual csv lines read: ',actualLineCount)
print('Count after Data Augumentation I : ',len(image_metadata))


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
get_ipython().magic('matplotlib inline')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#Split data into train and validation

image_metadata = shuffle(image_metadata)
train_samples,validation_samples = train_test_split(image_metadata,test_size=0.2,random_state=2)

print("Train_samples: ",len(train_samples))
print("Valid_samples: ",len(validation_samples))


# Data AUgumentation II
# each generator call read a batch from image_metadata
# For each line read, I will generate 6 images and corresponding angles
# left, center and right image. use correction_factor for left and right angles
# flip above 3 images and use corresponding negative angle
# this way, each run of generator of batch size 30 sends 180 images to model to train

def getBatchData(image_metadata,Batch_Size,fetch_side_camera_data=False):
    numSamples = len(image_metadata)
    while True:
        image_metadata = shuffle(image_metadata)
        for offset in range(0,numSamples,Batch_Size):
            batch_meta = image_metadata[offset : offset+Batch_Size]
            image_data=[]
            image_labels=[]
            for line in batch_meta:
                angle = float(line[3])
                correction_factor = 0.175
                if fetch_side_camera_data:
                    image_index = [0,1,2]
                    angle_index = [angle,angle+correction_factor,angle-correction_factor]
                else:
                    image_index = [0]
                    angle_index = [angle]
                for index,steer_angle in zip(image_index,angle_index):
                    image = os.path.join(dataDirectory,os.path.join(line[index].split(os.path.sep)[-2],line[index].split(os.path.sep)[-1]))
                    image_array = plt.imread(image)
                    image_data.append(image_array)
                    image_labels.append(steer_angle)
                    
                    flipped_image = cv2.flip(image_array,1)
                    image_data.append(flipped_image)
                    image_labels.append(steer_angle * -1.0)
            image_data = np.array(image_data)
            image_labels = np.array(image_labels)
            yield shuffle(image_data,image_labels)
            



batch_size = 30
train_generator = getBatchData(train_samples,Batch_Size=batch_size,fetch_side_camera_data=True)
validation_generator = getBatchData(validation_samples,Batch_Size=batch_size,fetch_side_camera_data=True)



#Keras Imports
from keras.layers import Input,Flatten,Dense,Dropout,Activation,Lambda
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.initializers import RandomNormal, Constant
import tensorflow as tf
tf.python.control_flow_ops = tf

#Keras Initializations
initialize_weights = RandomNormal(mean=0.0, stddev=0.1, seed=None)
initialize_bias = Constant(value=0.1)

#Keras Model
model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)),input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))

model.add(Conv2D(32,(3,3),strides=(1, 1),padding='same',
                 activation='relu',use_bias=True,kernel_initializer=initialize_weights,
                 bias_initializer=initialize_bias))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same',
                 activation='relu',use_bias=True,kernel_initializer=initialize_weights,
                 bias_initializer=initialize_bias))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128,(3,3),strides=(1, 1),padding='same',
                 activation='relu',use_bias=True,kernel_initializer=initialize_weights,
                 bias_initializer=initialize_bias))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128,(3,3),strides=(1, 1),padding='same',
                 activation='relu',use_bias=True,kernel_initializer=initialize_weights,
                 bias_initializer=initialize_bias))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, validation_data=validation_generator,verbose=1, validation_steps=math.ceil(len(validation_samples) / batch_size), epochs=7,steps_per_epoch = math.ceil(len(train_samples) / batch_size))


model.save('model.h5')

