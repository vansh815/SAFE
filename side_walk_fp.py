import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from PIL import Image
import random
import time
import pickle

model = tf.keras.models.load_model('sidewalk_model.h5')

"""
cnt = 0
prev = time.time()
for file in os.listdir("IU_image"):
  image = Image.open("IU_image/"+file)
  image = np.array(image)
  recon_image = []
  for i in range(3):
    recon_image.append(image[..., i].transpose().reshape(640, 300, 1))
  image = np.concatenate(recon_image, axis=2)
  image = image.reshape((1, 640, 300, 3))
  image = image / 255.
  prediction = model.predict(image)
  cnt += 1
  if cnt == 10000:
    break
    
print(time.time()-prev)
"""
traindata_gen = ImageDataGenerator(rescale = 1.0/255.)
test_generator = traindata_gen.flow_from_directory('IU_images', class_mode='categorical', target_size=(640, 300), batch_size = 32, shuffle = False)
print(test_generator.filenames[0][7:])

predIdxs_train = model.predict_generator(test_generator)

final_dict_zebra = {}
for x,y in zip(test_generator.filenames, predIdxs_train):
  final_dict_zebra[x[7:]] = y

with open('sidewalk_result', 'wb') as handle:
  pickle.dump(final_dict_zebra, handle)
