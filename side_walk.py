import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

traindata_gen = ImageDataGenerator(rescale = 1.0/255., brightness_range=[0.2, 1.0], horizontal_flip=True, zoom_range=[0.5, 1.0])
testdata_gen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = traindata_gen.flow_from_directory('/N/slate/jp109/sidewalk_data/train/', class_mode='categorical', shuffle = True, target_size=(640, 300))
test_generator = traindata_gen.flow_from_directory('/N/slate/jp109/sidewalk_data/test/', class_mode='categorical', shuffle = False, target_size=(640, 300))


def res_block(inputs, nodes, filter_size):
    x = tf.keras.layers.Conv2D(nodes, (filter_size, filter_size), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(nodes, (filter_size, filter_size), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation('relu')(x)

    return x

RES_N = 12
inputs = tf.keras.Input(shape=(640,300,3))
x = tf.keras.layers.Conv2D(32, (11,11), strides = 4, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((3,3))(x)
x = tf.keras.layers.Conv2D(32, (3, 3))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((3,3))(x)

for i in range(RES_N):
    x = res_block(x, 64, 3)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0002, momentum=0.9, nesterov=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 3)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model_final = model.fit_generator(train_generator, epochs = 500, callbacks=[callback])

model.save('sidewalk_model.h5')

test_loss, test_acc, test_pre, test_rec = model.evaluate(test_generator)
print("loss, acc, precision, recall")
print(test_loss, test_acc, test_pre, test_rec)

predIdxs = model.predict(test_generator)
predicted = np.argmax(predIdxs, axis = 1)

print(classification_report(list(test_generator.classes), list(predicted)))
