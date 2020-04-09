#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
import tqdm

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.initializers import glorot_uniform



import matplotlib.pyplot as plt


# In[ ]:


train_path = '/kaggle/input/leaf-classification/dataset/train'
test_path = '/kaggle/input/leaf-classification/dataset/test'


# In[ ]:


img_channels = 3
nb_classes = 185
batch_size = 64


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#height_shift_range=[-30,1], horizontal_flip=True, rotation_range=30, vertical_flip = True
train_datagenerator = train_datagen.flow_from_directory(train_path,
                                                      target_size = (224,224),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical',
                                                      shuffle = True)

validation_generator = test_datagen.flow_from_directory(test_path,
                                                       target_size = (224,224),
                                                       shuffle = True,
                                                       class_mode = 'categorical')


# In[ ]:


STEP_SIZE_TRAIN=train_datagenerator.n//train_datagenerator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size


# Identity block

# In[ ]:


def block_1(X, f, filters, s = 2):

    F1,F2,F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    #shortcut 
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


def block_2(X, f, filters):
    
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


# ResNet50

# In[ ]:


def ResNet50(input_shape=(224,224, 3), classes = 185):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = block_1(X, f=3, filters=[64, 64, 256], s=1)
    X = block_2(X, 3, [64, 64, 256])
    X = Dropout(0.2)(X)
    X = block_2(X, 3, [64, 64, 256])

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = block_1(X, f = 3, filters = [128, 128, 512],s = 2)
    X = block_2(X, 3, [128, 128, 512])
    X = Dropout(0.2)(X)
    X = block_2(X, 3, [128, 128, 512])
    X = block_2(X, 3, [128, 128, 512])

    # Stage 4 (≈6 lines)
    X = block_1(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = Dropout(0.2)(X)
    X = block_2(X, 3, [256, 256, 1024])
    X = block_2(X, 3, [256, 256, 1024])
    X = block_2(X, 3, [256, 256, 1024])
    X = Dropout(0.2)(X)
    X = block_2(X, 3, [256, 256, 1024])
    X = block_2(X, 3, [256, 256, 1024])
    X = Dropout(0.2)(X)

    # Stage 5 (≈3 lines)
    X = block_1(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = block_2(X, 3, [512, 512, 2048])
    X = block_2(X, 3, [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2))(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dropout(0.2)(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


# In[ ]:


model = ResNet50(input_shape = (224, 224, 3), classes = 185)


# In[ ]:


for layer in model.layers:
    layer.trainable = True


# In[ ]:


es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='model.h5',monitor = 'val_loss', verbose=1, save_best_only=True)

#Lr = CyclicLR(base_lr = 0.00001, max_lr = 0.0001, step_size = 2 * STEP_SIZE_TRAIN, mode = 'traingular2')


# In[ ]:


#compile model using accuracy to measure model performance
model.compile(optimizer = keras.optimizers.Adam(lr = 0.001),
              loss='categorical_crossentropy', metrics=['accuracy',])

history = model.fit_generator(generator=train_datagenerator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30,
                             callbacks = [es])


# In[ ]:


#compile model using accuracy to measure model performance
model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001),
              loss='categorical_crossentropy', metrics=['accuracy',])

history = model.fit_generator(generator=train_datagenerator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25,
                             callbacks = [es, checkpointer])

