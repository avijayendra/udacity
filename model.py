import os
import csv
import math

samples = []
with open('./data/driving_log.csv') as csvfile:
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
                centername = './data/IMG/'+batch_sample[0].split('/')[-1]
                leftname = './data/IMG/'+batch_sample[1].split('/')[-1]
                rightname = './data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(centername)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Left Images
                angle_offset = float(0.2) # To be experimented
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
model.add(Activation('softmax'))

model.summary()

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

model.compile(loss='mse', optimizer='adam')

# Train
# Since center, left and right images are used, number of samples becomes 3 times input
model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(3*train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(3*validation_samples)/batch_size),
            epochs=3, verbose=3)

# Save model
model.save('model.h5')