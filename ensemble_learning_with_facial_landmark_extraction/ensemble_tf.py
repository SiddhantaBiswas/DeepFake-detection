# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:30:45 2021

@author: Siddhanta Biswas
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import Model as KerasModel
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU, Embedding, LSTM
from keras.optimizers import Adam
import cv2
import extract_landmarks


BATCH_SIZE = 25
SIZE = 100
TRAINING_EPOCHS = 2
TUNING_EPOCHS = 15

input_shape = (SIZE, SIZE, 3)
learning_rate = 0.001
tuning_learning_rate = 0.00002

directory = r"C:\Users\Siddhanta Biswas\Desktop\Face Detection and Extraction\test\testing"
image_format = '.jpg'

# When shuffle = False, set seed = None (default value)
# When shuffle = false and seed = random int value, 
# then a single image is extracted multiple times
training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(SIZE, SIZE),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=9,
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(SIZE, SIZE),
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=2,
)

# printing class names of dataset
class_names = training_dataset.class_names
print(class_names)

# Printing dataset sizes
# training_dataset contains 576 files in 25 batches.
# len(training_dataset) = 576 / 25 = 23.04 = 24
# validation_dataset contains 144 files in 25 batches
# len(validation_dataset) = 144 / 25 = 5.76 = 6
print('Dataset Sizes:')
print(len(training_dataset))
print(len(validation_dataset))

# We can enumerate each batch by using either Python’s enumerator or a build-in method. 
# Enumerator enumerate() produces a tensor, which is recommended.
print('Printing shapes off all the batches by enumerating over the dataset')
for index, batch in validation_dataset.enumerate():
    print(index, len(batch))
 
# If you do not need a whole dataset, you can take a desired number of batches from it.  
# take() method creates a Dataset with at most 'n' elements from this dataset. Here n = 1
# So 1 batch of BATCH_SIZE number of images are extracted.
# Each 'batch' is a tuple of tf.Tensor of size = 2.
# First dimension contains the image arrays of all images
# Second dimension contains the labels for all corresponding images
print('Printing structure of an individual batch of images')
for batch in validation_dataset.take(1):
    print(len(batch))
    # print(batch[0])
    print(batch[1]) 
  
#=============================================================================   
  
def plot_dataset_1():
    # Printing all images in the validation dataset batch by batch using take(n)
    # n = len(validation_dataset) will give all batches in validation set
    print('Plotting all batches of images from validation set...')
    index = 0
    for images, labels in validation_dataset.take(len(validation_dataset)):
        index += 1
        print(labels)
        fig = plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        fig.suptitle('Batch ' + str(index))
    plt.show()
#plot_dataset_1()

def plot_dataset_2():    
    # Printing all images in the validation dataset batch by batch using enumerate()    
    print('Plotting all batches of images from validation set...')
    for index, batch in validation_dataset.enumerate():
        print(index, len(batch))
        print('Image Tensor shape: ' + str(batch[0].shape))
        print('Label Tensor shape: ' + str(batch[1].shape))
        print('\n')
        plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(batch[0][i].numpy().astype("uint8"))
            plt.title(class_names[batch[1][i]])
            plt.axis("off")
        plt.show()
#plot_dataset_2()

def plot_dataset_3():
    # Printing all images in the validation dataset batch by batch using __iter__() 
    # Returns an tf.data.Iterator for the elements of this dataset.
    print('Plotting all batches of images from validation set...')
    iterator = validation_dataset.__iter__()
    while True:
        batch = iterator.get_next()
        plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(batch[0][i].numpy().astype("uint8"))
            plt.title(class_names[batch[1][i].numpy()])
            plt.axis("off")
        plt.show()
#plot_dataset_3()

def plot_dataset_4():
    # Printing all images in the validation dataset batch by batch using as_numpy_iterator()    
    # Returns an iterable over the elements of the dataset, with their tensors converted to numpy arrays.
    # Iterates infinitely over the dataset
    # When dataset is set to shuffle = False, only a single continuous batch of images is printed everytime.
    print('Plotting all batches of images from validation set...')
    while True:
        image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
        plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis("off")
        plt.show()
#plot_dataset_4()



# Returns a <generator object custom_training_generator>
'''
def custom_training_generator():
    for index, batch in training_dataset.enumerate():
        images_channel_1 = batch[0]
        labels = batch[1]
        images_channel_2 = []
        for image in images_channel_1:
            image = cv2.resize(image.numpy(), (10,10))
            images_channel_2.append(image)
            
        images_channel_2 = tf.convert_to_tensor(images_channel_2, dtype=None)
        yield [images_channel_1, images_channel_2], labels
'''
def custom_training_generator():
    while True:
        image_batch, label_batch = training_dataset.as_numpy_iterator().next()
        images_channel_1 = image_batch
        labels = label_batch
        eyes, lips, noses = extract_landmarks.process_batch(image_batch)
        images_channel_2 = []
        # for image in images_channel_1:
        #     image = cv2.resize(image, (10,10))
        #     images_channel_2.append(image)
            
        eyes = tf.keras.applications.xception.preprocess_input(eyes)
        eyes = tf.convert_to_tensor(eyes, dtype=None)
        # images_channel_2 = tf.convert_to_tensor(images_channel_2, dtype=None)
        yield [images_channel_1, eyes], labels
        
# Returns a <generator object custom_validation_generator>
def custom_validation_generator():
    while True:
        image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
        images_channel_1 = image_batch
        labels = label_batch
        eyes, lips, noses = extract_landmarks.process_batch(image_batch)
        images_channel_2 = []
        # for image in images_channel_1:
        #     image = cv2.resize(image, (10,10))
        #     images_channel_2.append(image)
            
        eyes = tf.keras.applications.xception.preprocess_input(eyes)
        eyes = tf.convert_to_tensor(eyes, dtype=None)
        # images_channel_2 = tf.convert_to_tensor(images_channel_2, dtype=None)
        yield [images_channel_1, eyes], labels
        
#=============================================================================

def channel(input_shape, inputs):
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(inputs)
        x1 = BatchNormalization()(x1)
        y1 = Flatten()(x1)
        y1 = Dropout(0.5)(y1)
        y1 = Dense(16)(y1)
        y1 = LeakyReLU(alpha=0.1)(y1)

        return y1
    
def multichannel():

 	# channel 1
    input1 = Input(shape = (SIZE, SIZE, 3)) 
    input2 = Input(shape = (300, 100, 3))
  
    y1 = channel(input_shape, input1)
    y2 = channel((300,100,3), input2)

    # merge
    merged = concatenate([y1, y2])

    # interpretation
    dense1 = Dense(16)(merged)
    dense1 = LeakyReLU(alpha=0.1)(dense1)
    dense1 = Dropout(0.5)(dense1)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[input1, input2], outputs=outputs)

    return model, 'multichannel'

model, modelName = multichannel()
optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()

#=============================================================================

history = model.fit(custom_training_generator(),
                    epochs=TRAINING_EPOCHS,
                    validation_data=custom_validation_generator(),
                    steps_per_epoch=len(training_dataset),
                    validation_steps=len(validation_dataset),
                    )

#=============================================================================

def testchannel(inputs):
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(inputs)
        return x1

def testmodel():
    input1 = Input(shape = (SIZE, SIZE, 3)) 
    input2 = Input(shape = (300, 100, 3))
 
    y1 = testchannel(input1)
    y2 = testchannel(input2)
    
    # merged = concatenate([y1, y2])

    # dense1 = Dense(16)(merged)
    # dense1 = LeakyReLU(alpha=0.1)(dense1)
    # dense1 = Dropout(0.5)(dense1)
    # outputs = Dense(1, activation='sigmoid')(dense1)
    # test_model = Model(inputs=[input1, input2], outputs=outputs)

    model = Model(inputs=[input1, input2], outputs=[y1, y2])
    return model, 'testmodel'

test_model, modelName = testmodel()
optimizer = Adam(lr = 0.001)
test_model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
test_model.summary()

#=============================================================================

filters, biases = test_model.layers[2].get_weights()
predictions = test_model.predict(custom_validation_generator(), steps=1)
print(len(predictions[1]))


# Visualizing feature extraction from intermediate conv2D layers   
cols = 4
rows = 4
for count in range(0, len(predictions[1])):
    fig = plt.figure(figsize=(7, 7))
    for i in range(1, 8+1):
        fig = plt.subplot(rows, cols, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(predictions[0][count][:,:,i-1], cmap='gray')
        #break
    plt.show()
    for i in range(1, 8+1):
        fig = plt.subplot(rows, cols, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(predictions[1][count][:,:,i-1], cmap='gray')
        #break
    plt.show()





