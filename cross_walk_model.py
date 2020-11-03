import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from numpy.random import seed
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.regularizers import l2

tf.compat.v1.set_random_seed(1)
seed(1)
traindata_gen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = traindata_gen.flow_from_directory('train/', class_mode='categorical', shuffle = True, target_size=(640, 300), batch_size = 32)

validation_generator = traindata_gen.flow_from_directory('test/', class_mode='categorical', target_size=(640, 300), shuffle = False, batch_size = 32)

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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])

model_final = model.fit(train_generator, validation_data= validation_generator, steps_per_epoch=50, epochs = 15)

model.save('cross_walk_model.h5')

# Test prediction
predIdxs = model.predict(validation_generator)
predicted = np.argmax(predIdxs, axis = 1)

# Validation performance
print(classification_report(list(validation_generator.classes), list(predicted)))
