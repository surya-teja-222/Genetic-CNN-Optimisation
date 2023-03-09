import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.stderr = open('SVCerror.txt', 'w')
sys.stdout = open('SVCoutput.txt', 'w')

# +
# def loadImages(path, urls,target ):
#   images = []
#   labels = []
#   #for i in range(len(urls))
#   for i in range(len(urls)):
#     img_path = path + "/" + urls[i]
#     img = cv2.imread(img_path)
#     img = img / 255.0
#     # if we want to resize the images
#     img = cv2.resize(img, (100, 100))
#     images.append(img)
#     labels.append(target)
#   images = np.asarray(images)
#   return images, labels

# +
# covid_path = "./../data/COVID-19_Radiography_Dataset/COVID/images"
# covidUrl = os.listdir(covid_path)
# covidImages, covidTargets = loadImages(covid_path, covidUrl, 1)

# +
# normal_path = "./../data/COVID-19_Radiography_Dataset/Normal/images"
# normal_urls = os.listdir(normal_path)
# normalImages, normalTargets = loadImages(normal_path, normal_urls, 0)

# +
# covidImages=np.asarray(covidImages)

# +
# normalImages=np.asarray(normalImages)

# +
# data = np.r_[covidImages, normalImages]

# +
# targets = np.r_[covidTargets, normalTargets]
# -



# +
# x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)
# -

from test_train_data import x_train, x_test, y_train, y_test

# flatten the images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# train the model
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)

# evaluate the model
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# +
# find accuracy, precision, recall, f1 score, jaccard index, kappa score, confusion matrix, ROC curve, AUC score, etc.

# precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='macro')
print('Precision: %f' % precision)

# recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='macro')
print('Recall: %f' % recall)

# f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro')
print('F1 score: %f' % f1)

# jaccard index
from sklearn.metrics import jaccard_score
jaccard = jaccard_score(y_test, y_pred, average='macro')
print('Jaccard score: %f' % jaccard)

# kappa score
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test, y_pred)
print('Cohens kappa: %f' % kappa)

# confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(fpr, tpr, thresholds)

# AUC score
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print('ROC AUC=%.3f' % (auc))

# +

# roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# calculate roc curves
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.', label='ROC')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
# -
