import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split


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

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)

def getTestTrainData():
    return x_train, y_train, x_test, y_test