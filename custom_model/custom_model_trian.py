import numpy as np
import os
import cv2
import imageio

##Path of the dataset
train_path='../input/leaf-classification/dataset/train'
test_path='../input/leaf-classification/dataset/test'


X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

for i in os.listdir(train_path):
  print(i)
  for j in os.listdir(train_path + '/' + i):
    img=imageio.imread(train_path+'/'+i+'/'+j)
    img=cv2.resize(img, (64, 64))
    X_train.append(img)
    Y_train.append(i)
  for j in os.listdir(test_path + '/' + i):
    img=imageio.imread(test_path+'/'+i+'/'+j)
    img=cv2.resize(img, (64, 64))
    X_test.append(img)
    Y_test.append(i)
    
X_train=np.stack(X_train)
X_train=X_train/255
Y_train=np.stack(Y_train)
X_test=np.stack(X_test)
X_test=X_test/255
Y_test=np.stack(Y_test)


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_train = Y_train.reshape(len(Y_train), 1)
Y_test = label_encoder.fit_transform(Y_test)
Y_test = Y_test.reshape(len(Y_test), 1)

onehot_encoder = OneHotEncoder(sparse=False)
Y_train = onehot_encoder.fit_transform(Y_train)
Y_test = onehot_encoder.fit_transform(Y_test)

from sklearn.utils import shuffle

for i in range(1, 5):
    Y_train, X_train=shuffle(Y_train, X_train)
    Y_test, X_test=shuffle(Y_test, X_test)
 


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint



model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Dense(185, activation='softmax'))


model.compile(optimizer=keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, decay=1e-8, amsgrad=False),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_file = "leaf_reco.h5"
model_1 = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(X_train, Y_train, epochs=400, batch_size=64, validation_data=(X_test, Y_test), callbacks=[model_1])
