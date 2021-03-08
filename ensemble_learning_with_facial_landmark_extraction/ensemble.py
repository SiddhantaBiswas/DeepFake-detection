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

BATCH_SIZE = 10
SIZE = 100
TRAINING_EPOCHS = 5
TUNING_EPOCHS = 15

input_shape = (SIZE, SIZE, 3)
learning_rate = 0.001
tuning_learning_rate = 0.00002

directory = os.path.join('test', 'testing')
image_format = '.jpg'

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #preprocessing_function=tf.keras.applications.xception.preprocess_input ,
    validation_split=0.2,
)

training_dataset = generator.flow_from_directory(
    directory,
    target_size=(SIZE, SIZE),
    class_mode="binary",
    shuffle=True,
    seed=42,
    batch_size=BATCH_SIZE,
    subset="training",
)
validation_dataset = generator.flow_from_directory(
    directory,
    target_size=(SIZE, SIZE),
    class_mode="binary",
    shuffle=True,
    seed=42,
    batch_size=BATCH_SIZE,
    subset="validation",
)

# =============================================================================
# def plot(dataset_generator):
#     labels_dict = dataset_generator.class_indices
#     labels_list = list(labels_dict.keys())
#     
#     plt.figure(figsize=(7, 7))
#     images, labels = next(dataset_generator)
#     
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i] / 2 + 0.5)
#         label_index = int(labels[i])
#         plt.title(labels_list[label_index])
#         plt.axis("off")
#         
# #plot(training_dataset)
# #plot(validation_dataset)
# =============================================================================

def custom_training_generator(directory=directory, batch_size=BATCH_SIZE):
    while True:
        X1i = training_dataset.next()
        temp = []
        images = []
        labels = []
        for img in X1i[0]:
            img = cv2.resize(img, (SIZE, SIZE))
            images.append(img)
        for label in X1i[1]:
            labels.append(label)
        images = np.array(images)
        images = tf.keras.applications.xception.preprocess_input(images)
        temp.append(images)
        temp.append(np.array(labels))
        X2i = tuple(temp)
        
        yield [X1i[0], X2i[0]], X1i[1]
        
def custom_validation_generator(directory=directory, batch_size=BATCH_SIZE):
    while True:
        X1i = training_dataset.next()
        temp = []
        images = []
        labels = []
        for img in X1i[0]:
            img = cv2.resize(img, (SIZE, SIZE))
            images.append(img)
        for label in X1i[1]:
            labels.append(label)
        images = np.array(images)
        images = tf.keras.applications.xception.preprocess_input(images)
        temp.append(images)
        temp.append(np.array(labels))
        X2i = tuple(temp)
        
        yield [X1i[0], X2i[0]], X1i[1]
        

# =============================================================================
# gen = custom_training_generator(directory)
# for images, labels in gen:
#     print(labels[0])
#     print(labels[1])
#     break
# =============================================================================


     
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
    input2 = Input(shape = (SIZE, SIZE, 3))
 
    y1 = channel(input_shape, input1)
    y2 = channel(input_shape, input2)

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
optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()
print(modelName)

history = model.fit(custom_training_generator(),
                    epochs=TRAINING_EPOCHS,
                    validation_data=custom_validation_generator(),
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=len(training_dataset),
                    validation_steps=len(validation_dataset),
                    shuffle=True)
