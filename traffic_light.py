#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.regularizers import l2




traindata_gen = ImageDataGenerator(rescale = 1.0/255.)
#traindata_gen = ImageDataGenerator(
 #       rescale=1.0/255.,
  #      shear_range=0.2,
   #     zoom_range=0.2,
    #    horizontal_flip=False)

testdata_gen = ImageDataGenerator(rescale = 1.0/255.)
train_generator = traindata_gen.flow_from_directory('/Users/vanshsmacpro/Downloads/dataset_t/train/', 
                                                    class_mode='categorical', shuffle = True,
                                                    target_size=(640, 300),
                                                    batch_size = 32)

test_generator = testdata_gen.flow_from_directory('/Users/vanshsmacpro/Downloads/dataset_t/test/',
                                                  class_mode='categorical', shuffle = True
                                                  , target_size=(640, 300), batch_size = 32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11,11), activation = 'relu', input_shape = (640,300,3), strides = 4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3,3), strides=2),
    tf.keras.layers.Conv2D(256, (5,5), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3,3), strides=2),
    tf.keras.layers.Conv2D(384, (3,3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(384, (3,3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3,3), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9216, activation = 'relu'),
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000001)

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['acc'])

model_final = model.fit(train_generator, validation_data= test_generator ,epochs = 5, 
                        steps_per_epoch=len(train_generator) , validation_steps= len(test_generator))


predIdxs = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
#predIdxs = model.predict_generator(validate, steps=len(validate), verbose=1)
predicted = np.argmax(predIdxs, axis = 1)
print(classification_report(list(test_generator.classes), list(predicted)))


# Train Prediction
#predIdxs_train = model.predict_generator(train_generator)
#predicted_train = np.argmax(predIdxs_train, axis = 1)
model = tf.keras.models.load_model('/Users/vanshsmacpro/Desktop/stop_sign3.h5')

validate = testdata_gen.flow_from_directory('/Users/vanshsmacpro/Desktop/ComputerVision/validate/', 
                                            class_mode='categorical',
                                            target_size=(640, 300), shuffle = False,batch_size = 32)
predIdxs = model.predict_generator(validate, steps=len(validate), verbose=1)
predicted = np.argmax(predIdxs, axis = 1)
print(predicted)
print(classification_report(list(validate.classes), list(predicted)))
model.save('/Users/vanshsmacpro/Desktop/stop_sign2.h5')
model = tf.keras.models.load_model('/Users/vanshsmacpro/Downloads/stop_sign_model/model_crossing_sign_pedestrian.h5')
predicted_different = []
for i in range(len(predIdxs)):

    if predIdxs[i][1] > 0.2 :
        predicted_different.append(1)
    else : 
        predicted_different.append(0)
        
#predIdxs = model.predict_generator(validate, steps=len(validate), verbose=1)
print(predIdxs)
#predicted = np.argmax(predIdxs, axis = 1)
#print(predicted_different)
print(classification_report(list(validate.classes), list(predicted_different)))




# Test prediction
