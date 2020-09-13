### Project: Behaviour Cloning
# Problem Statement: Build a Keras model to clone vehicle behaviour and keep the vehicle on track
# Input: Recorded data from simulator in training mode
# Output: Steering angle to keep the vehicle on the track
# Important Note: Do not use any activation function in the output layer, since the output is a continuous value, not binary
#
#
### Solution Design
# 1. Used training and validation generator to capture data and use of yield to hold the data
# 2. Used Keras to build network model
# 3. Used two approaches for neural network models described below.
# 4. Calculated output shape for each step (as it is new learning and need to figure out how it is calculated)
# 5. No data augmentation used here, however while capturing data, vehicle was driven in reverse direction
#
# Approach 1: Keras model starting with flattened images and then couple of fully-connected layers, output is steering value
# Network Architecture - Generated from model.summary() of a Keras Sequential model:
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
# _________________________________________________________________
# lambda_1 (Lambda)            (None, 80, 320, 3)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 76800)             0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               9830528   
# _________________________________________________________________
# activation_1 (Activation)    (None, 128)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 60)                7740      
# _________________________________________________________________
# activation_2 (Activation)    (None, 60)                0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 61        
# =================================================================
# Total params: 9,838,329
# Trainable params: 9,838,329
# Non-trainable params: 0
#
# Training date: Center, Left and Right images (with steering offsets 0.2
# Hyper parameters: Epoch 3
# Results: training without GPU
# Epoch 1/3
# 603/603 [==============================] - 404s 669ms/step - loss: 0.5120 - val_loss: 0.0385
# Epoch 2/3
# 603/603 [==============================] - 398s 660ms/step - loss: 0.0311 - val_loss: 0.0322
# Epoch 3/3
# 603/603 [==============================] - 397s 659ms/step - loss: 0.0254 - val_loss: 0.0359
# 
# Results: training with GPU 
# Epoch 1/3
# 603/603 [==============================] - 72s 119ms/step - loss: 0.3105 - val_loss: 0.0456
# Epoch 2/3
# 603/603 [==============================] - 70s 116ms/step - loss: 0.0297 - val_loss: 0.0459
# Epoch 3/3
# 603/603 [==============================] - 71s 118ms/step - loss: 0.0221 - val_loss: 0.0688
# 
# Approach 2: Use couple of CNN layers to analyze and train from the input images
# Network Architecture - Generated from model.summary() of a Keras Sequential model:
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
# _________________________________________________________________
# lambda_1 (Lambda)            (None, 80, 320, 3)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 78, 318, 32)       896       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 39, 159, 32)       0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 37, 157, 64)       18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 18, 78, 64)        0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 16, 76, 128)       73856     
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 8, 38, 128)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 38912)             0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               4980864   
# _________________________________________________________________
# activation_1 (Activation)    (None, 128)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 60)                7740      
# _________________________________________________________________
# activation_2 (Activation)    (None, 60)                0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 61        
# =================================================================
# Total params: 5,081,913
# Trainable params: 5,081,913
# Non-trainable params: 0
# _________________________________________________________________
#
# Output: (Sample Data from Project resources)
# 2020-09-13 12:31:28.780711: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-09-13 12:31:28.782882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 0000:00:04.0
# Total memory: 11.17GiB
# Free memory: 11.10GiB
# Epoch 1/3
# 603/603 [==============================] - 203s 337ms/step - loss: 0.0296 - val_loss: 0.0248
# Epoch 2/3
# 603/603 [==============================] - 200s 332ms/step - loss: 0.0205 - val_loss: 0.0234
# Epoch 3/3
# 603/603 [==============================] - 192s 318ms/step - loss: 0.0154 - val_loss: 0.0226
#
# Output: (Track 1 data)
# Epoch 1/3
# 401/401 [==============================] - 131s 326ms/step - loss: 0.0580 - val_loss: 0.0549
# Epoch 2/3
# 401/401 [==============================] - 125s 312ms/step - loss: 0.0402 - val_loss: 0.0433
# Epoch 3/3
# 401/401 [==============================] - 125s 311ms/step - loss: 0.0283 - val_loss: 0.0384
#
# Output: (Track 2 data)
# 436/436 [==============================] - 154s 353ms/step - loss: 0.1565 - val_loss: 0.1289
# Epoch 2/3
# 436/436 [==============================] - 137s 315ms/step - loss: 0.0835 - val_loss: 0.1192
# Epoch 3/3
# 436/436 [==============================] - 136s 311ms/step - loss: 0.0505 - val_loss: 0.1113
#
# Output: (Combined data - Track 1 and Track 2 data combined)
# Epoch 1/3
# 836/836 [==============================] - 267s 319ms/step - loss: 0.1098 - val_loss: 0.0981
# Epoch 2/3
# 836/836 [==============================] - 262s 314ms/step - loss: 0.0683 - val_loss: 0.0859
# Epoch 3/3
# 836/836 [==============================] - 261s 313ms/step - loss: 0.0385 - val_loss: 0.0786
#
### Project Rubrics
# Required Files
# model.py in the base folder (/home/workspace)
# Under CarND-Behavioral-Cloning-P3 folder:
# 1. sample_data - model.h5 files for different models without CNN and with CNN (without and with GPU)
# 2. track1_data - model.h5 file for track 1 (new data recorded - 3 tracks normal, 0.5 track reverse direction)
# 3. track2_data - model.h5 file for track 2 (new data recorded - 2 tracks normal)
# 4. track1_run - output_video.mp4 file for track 1 as input data and run in autonomous mode
# 5. combined_run - output_video.mp4 file for combined track as input data and run in autonomous mode
#
# Functional code - Yes
# Readable code - Yes
# Model architecture - two approaches with training results captured above 
# Overfitting corrections - yes, Dropouts were added, but then removed, since the model accuracy is around 97-98%
# Model tuning - epochs were adjusted to be lower since higher epochs were not reducing the loss much
# Training data - used sample data provided with project resource to start with. Also captured new data from both track 1 & track 2
# Solution design - documented above
# Model architecture - documented above
# Process for creating training data and process documented - documented above
# Car navigation on track 1 - TODO: currently it stays on course until track is roughly straight, but goes out of track on sharp turns
# I can improve it later on, after fixing other pending projects.

import os
import csv
import math

samples = []
with open('./combined/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0
    for line in reader:
        if (counter == 0):
            print("Header line:", line)
        else:
            samples.append(line)
        counter += 1
        
    print("number of records", counter)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                centername = './combined/IMG/'+batch_sample[0].split('/')[-1]
                leftname = './combined/IMG/'+batch_sample[1].split('/')[-1]
                rightname = './combined/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(centername)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Left Images
                angle_offset = float(0.1) # To be experimented, 0.2 landed the vehicle in water (right side)
                left_image = cv2.imread(leftname)
                left_angle = float(batch_sample[3]) - angle_offset
                images.append(left_image)
                angles.append(left_angle)
                
                # Right Images
                right_image = cv2.imread(rightname)
                right_angle = float(batch_sample[3]) + angle_offset
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import keras

# from keras import backend as K
# if K.image_data_format() == 'channels_first':
#     print("channel first")
#     input_shape = (ch, row, col)
# else:
#     print("channel last")
#     input_shape = (row, col, ch)

num_classes = 1
ch, row, col = 3, 160, 320  # Input image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# Crop 50 pixels from top and 30 pixels from bottom
# Input 160x320 Output 80x320
y_crop = (50, 30)
x_crop = (0, 0)
model.add(Cropping2D(cropping=((50,30), (0,0)), input_shape=(row,col,ch)))
row = 160 - np.sum(y_crop)
col = 320 - np.sum(x_crop)

# Lambda layer to normalize data and center it to mean
model.add(Lambda(lambda x: (x/255.0) - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

#model.add(... finish defining the rest of your model architecture here ...)
# Input 80x320x3, Output 78x318x32
kernel_size = (3, 3)
padding = (0, 0)
stride = (1, 1)
out_ch = 32
model.add(Conv2D(out_ch, kernel_size=kernel_size,
                 activation='relu',
                 input_shape=[row,col,ch],
                 padding='VALID'))
row = ((row - kernel_size[0] + 2 * padding[0])/(stride[0])) + 1
col = ((col - kernel_size[1] + 2 * padding[1])/(stride[1])) + 1
ch = out_ch
print("Conv output", row, col, ch)

# Input 76x316x32, Output 25x105,32 74x314x3(wrong)
# output_shape = (input_shape - pool_size + 1) / strides)
pool_size = (2, 2)
stride = (2, 2)
model.add(MaxPooling2D(pool_size=pool_size))
row = math.ceil(((row - pool_size[0] + 1)/(stride[0])))
col = math.ceil(((col - pool_size[1] + 1)/(stride[1])))
print("Maxpool output", row, col, ch)
#model.add(Dropout(0.5))

# Input 80x320x3, Output 76x316x32
kernel_size = (3, 3)
padding = (0, 0)
stride = (1, 1)
out_ch = 64
model.add(Conv2D(out_ch, kernel_size=kernel_size,
                 activation='relu',
                 input_shape=[row,col,ch],
                 padding='VALID'))
row = ((row - kernel_size[0] + 2 * padding[0])/(stride[0])) + 1
col = ((col - kernel_size[1] + 2 * padding[1])/(stride[1])) + 1
ch = out_ch
print("Conv output", row, col, ch)

# Input 76x316x32, Output 25x105,32 74x314x3(wrong)
# output_shape = (input_shape - pool_size + 1) / strides)
pool_size = (2, 2)
stride = (2, 2)
model.add(MaxPooling2D(pool_size=pool_size))
row = math.ceil(((row - pool_size[0] + 1)/(stride[0])))
col = math.ceil(((col - pool_size[1] + 1)/(stride[1])))
print("Maxpool output", row, col, ch)
#model.add(Dropout(0.5))

# Input 80x320x3, Output 76x316x32
kernel_size = (3, 3)
padding = (0, 0)
stride = (1, 1)
out_ch = 128
model.add(Conv2D(out_ch, kernel_size=kernel_size,
                 activation='relu',
                 input_shape=[row,col,ch],
                 padding='VALID'))
row = ((row - kernel_size[0] + 2 * padding[0])/(stride[0])) + 1
col = ((col - kernel_size[1] + 2 * padding[1])/(stride[1])) + 1
ch = out_ch
print("Conv output", row, col, ch)

# Input 76x316x32, Output 25x105,32 74x314x3(wrong)
# output_shape = (input_shape - pool_size + 1) / strides)
pool_size = (2, 2)
stride = (2, 2)
model.add(MaxPooling2D(pool_size=pool_size))
row = math.ceil(((row - pool_size[0] + 1)/(stride[0])))
col = math.ceil(((col - pool_size[1] + 1)/(stride[1])))
print("Maxpool output", row, col, ch)
#model.add(Dropout(0.2))

#1st Layer - Add a flatten layer
# Input 
# Output 25x105x32 = 84000
model.add(Flatten(input_shape=(row, col, ch)))

#2nd Layer - Add a fully connected layer
# Output 128
model.add(Dense(128))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
#model.add(Dropout(0.4))

#4th Layer - Add a fully connected layer
# Output 60
model.add(Dense(60))

#5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
#model.add(Dropout(0.6))

# Output num_classes
model.add(Dense(num_classes))
# No activation in the output layer, since it is a continuous value, not binary
# model.add(Activation('softmax'))

model.summary()

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

model.compile(loss='mse', optimizer='adam')

from workspace_utils import active_session
 
# Using utility to keep the workspace active, since the training takes time and the
# workspace may keep sleeping, losing all progress
with active_session():
    # Train
    # Since center, left and right images are used, number of samples becomes 3 times input
    model.fit_generator(train_generator,
                steps_per_epoch=math.ceil(len(3*train_samples)/batch_size),
                validation_data=validation_generator,
                validation_steps=math.ceil(len(3*validation_samples)/batch_size),
                epochs=3, verbose=1)

# Save model
model.save('model.h5')