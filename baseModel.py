
# write all output to a file
import sys
sys.stdout = open('logs/results/baseModelGen.txt', 'w')
sys.stderr = open('logs/results/error/baseModelGen-err.txt', 'w')


import numpy as np
import pandas as pd

import tensorflow as tf

# DETAILS OF THE GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num TPU Available: ", len(tf.config.experimental.list_physical_devices('TPU')))
print("Num Physical Devices Available: ", len(tf.config.experimental.list_physical_devices()))


from tensorflow import keras

# import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import os


def loadImages(path, urls,target ):
  images = []
  labels = []
  #for i in range(len(urls))
  for i in range(len(urls)):
    img_path = path + "/" + urls[i]
    img = cv2.imread(img_path)
    img = img / 255.0
    # if we want to resize the images
    img = cv2.resize(img, (100, 100))
    images.append(img)
    labels.append(target)
  images = np.asarray(images)
  return images, labels


covid_path = "data/COVID-19_Radiography_Dataset/COVID/images"
covidUrl = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidUrl, 1)

normal_path = "data/COVID-19_Radiography_Dataset/Normal/images"
normal_urls = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normal_urls, 0)

covidImages=np.asarray(covidImages)

normalImages=np.asarray(normalImages)

data = np.r_[covidImages, normalImages]

targets = np.r_[covidTargets, normalTargets]


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)

model = Sequential([

    Conv2D(64, 3,padding='same', input_shape=(100,100,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3,padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3,padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3,padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision'])

# model.fit(x_train, y_train,batch_size=32,epochs=10,validation_data=(x_test, y_test))
model.fit(x_train, y_train,epochs=15,validation_data=(x_test, y_test))

model.save('models/base_model.h5')

# evaluate the model
print(model.evaluate(x_train, y_train, verbose=0))
print(model.evaluate(x_test,y_test, verbose=0))