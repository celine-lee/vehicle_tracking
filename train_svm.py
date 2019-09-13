# THIS CODE WAS WRITTEN AND RUNS ON GOOGLE COLAB, IN ORDER TO TAKE ADVANTAGE OF THE GPU.
# WILL LIKELY NEED TO BE MODIFIED TO BE RUN LOCALLY.

from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/My Drive/Object Detection/object tracking based on HOG"

!pip install imutils

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from google.colab.patches import cv2_imshow

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()

%cd "training/vott-csv-export"

import csv

vehicles_features = []
nonvehicles_features = []
all_features_train = []
all_labels_train = []

# want to keep same aspect ratio for everything, so implementing some padding...
# desired aspect ratio: 4:3
with open('cars-export.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count >= 20:
            break
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            image = cv2.imread(row[0])
            y = int(float(row[2]))
            ymax = int(float(row[4]))
            x = int(float(row[1]))
            xmax = int(float(row[3]))
            width = xmax - x
            height = ymax - y
            ratiowidth = 4
            ratioheight = 3
            if (height * ratiowidth /ratioheight) < width: # need to pad pixels vertically
                pad = (width * ratioheight / ratiowidth) - height
                if y < pad / 2 :
                    pad = pad - y
                    y = 0
                else:
                    y = y - (pad/2)
                    pad = pad - (pad/2)
                
                if(ymax > image.shape[0] - pad - 1) :
                    if(y > pad - (image.shape[0] - 1 - ymax)) : #there is room to pad more on the y side
                        pad = pad - (image.shape[0] - 1 - ymax)
                        ymax = image.shape[0] - 1
                        y = y - pad
                        pad = 0
                    else : # if it doesn't fit, ignore this one entirely.
                        continue
                else:
                    ymax = ymax + pad
                    pad = 0
                print("new coordinates after vertical padding are ", x, ", ", y, ", ", xmax, ", ", ymax, " with new ratio: ", (xmax-x)/(ymax-y))
            else:
                if (height * ratiowidth / ratioheight) > width: # need to pad pixels horizontally
                    pad = (height * ratiowidth / ratioheight) - width
                    if(x < pad / 2) :
                        pad = pad - x
                        x = 0
                    else:
                        x = x - (pad/2)
                        pad = pad - (pad/2)

                    if(xmax > image.shape[1] - pad - 1) :
                        if(x > pad - (image.shape[1]-1-xmax)) :
                            pad = pad - (image.shape[1] - 1 - xmax)
                            xmax = image.shape[1] - 1
                            x = x - pad
                            pad = 0
                        else :
                            continue
                    else:
                      xmax = xmax + pad
                      pad = 0
                    #print("new coordinates after horizontal padding are ", x, ", ", y, ", ", xmax, ", ", ymax, " with new ratio: ", (xmax-x)/(ymax-y))
            y = int(y)
            ymax = int(ymax)
            x = int(x)
            xmax = int(xmax) 
            if row[5] == 'car' or row[5] == 'truck':
                car_rect = image[y:ymax, x:xmax].copy()
                car_rect = imutils.resize(car_rect, width=100)
                cv2_imshow(car_rect)
                computed = hog.compute(image)
                vehicles_features.append(computed)
                all_labels_train.append(1)
            else:
                noncar_rect = image[y:ymax, x:xmax].copy()
                noncar_rect = imutils.resize(noncar_rect, width=100)
                cv2_imshow(noncar_rect)
                computed = hog.compute(image)
                nonvehicles_features.append(computed)
                all_labels_train.append(0)
            line_count += 1
            
    print(f'Processed {line_count} lines.')
    
    
print(len(vehicles_features))
print(len(nonvehicles_features))
all_features_train = np.vstack((vehicles_features, nonvehicles_features))
#all_labels_train = np.hstack((all_labels_train, np.zeros(len(nonvehicles_features))))
all_labels_train = np.array(all_labels_train)

all_features_train = all_features_train.reshape(all_features_train.shape[0], -1)
all_labels_train = all_labels_train.reshape(all_labels_train.shape[0], -1)

print(len(all_features_train))
print(len(all_labels_train))

#train SVM
from sklearn import svm
from sklearn.model_selection import train_test_split
import random as rand

# Training and testing features...
vehicle_ft_train, vehicle_ft_test, vehicle_label_train, vehicle_label_test = train_test_split(all_features_train, all_labels_train, test_size = 0.2, random_state = rand.randint(1, 100))
svc = svm.SVC()
svc.fit(vehicle_ft_train, vehicle_label_train)
print("here")
accuracy = svc.score(vehicle_ft_test, vehicle_label_test)
print(accuracy) #results using 20 lines has been: 0.25, 0.75