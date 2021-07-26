import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import pickle
import sys

DATADIR = "../Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

def create_training_data(IMG_SIZE):

    training_data = []
    train_images = []
    train_images_half = []
    train_images_quarter = []
    train_labels = []
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        total = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                if(img_array.shape[0] >= 256 & img_array.shape[1] >= 256):
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
                    new_array_half = cv2.resize(img_array, (IMG_SIZE/2, IMG_SIZE/2)) / 255.0
                    new_array_quarter = cv2.resize(img_array, (IMG_SIZE/4, IMG_SIZE/4)) / 255.0
                    training_data.append([new_array, new_array_half, new_array_quarter, class_num])
                    total += 1
            except Exception as e:
                print("an image failed")
            
            if(total == 500):
                break
    
    random.shuffle(training_data)

    for x, y, z , w in training_data:
        train_images.append(x)
        train_images_half.append(y)
        train_images_quarter.append(z)
        train_labels.append(w)
    
    train_images = np.array(train_images)
    train_images_half = np.array(train_images)
    train_images_quarter = np.array(train_images)
    train_labels = np.array(train_labels).reshape(train_images.shape[0], 1)

    pickle_out = open("train_images" + str(IMG_SIZE) + ".pickle", "wb")
    pickle.dump(train_images, pickle_out)
    pickle_out.close()

    pickle_out = open("train_images" + str(IMG_SIZE/2) + ".pickle", "wb")
    pickle.dump(train_images_half, pickle_out)
    pickle_out.close()

    pickle_out = open("train_images" + str(IMG_SIZE/4) + ".pickle", "wb")
    pickle.dump(train_images_quarter, pickle_out)
    pickle_out.close()

    pickle_out = open("train_labels.pickle", "wb")
    pickle.dump(train_labels, pickle_out)
    pickle_out.close()

create_training_data(256)

def create_testing_data(IMG_SIZE):

    testing_data = []
    test_images = []
    test_images_half = []
    test_images_quarter = []
    test_labels = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        total = 0
        for img in list(reversed(os.listdir(path))):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                if(img_array.shape[0] >= 256 & img_array.shape[1] >= 256):
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
                    new_array_half = cv2.resize(img_array, (IMG_SIZE/2, IMG_SIZE/2)) / 255.0
                    new_array_quarter = cv2.resize(img_array, (IMG_SIZE/4, IMG_SIZE/4)) / 255.0
                    testing_data.append([new_array, new_array_half, new_array_quarter, class_num])
                    total +=1
            except Exception as e:
                print("an image failed")
            
            if(total == 100):
                break
    
    random.shuffle(testing_data)

    for x, y, z , w in testing_data:
        test_images.append(x)
        test_images_half.append(y)
        test_images_quarter.append(z)
        test_labels.append(w)

    test_images = np.array(test_images)
    test_images_half = np.array(test_images)
    test_images_quarter = np.array(test_images)
    test_labels = np.array(test_labels).reshape(test_images.shape[0], 1)

    pickle_out = open("test_images" + str(IMG_SIZE) + ".pickle", "wb")
    pickle.dump(test_images, pickle_out)
    pickle_out.close()

    pickle_out = open("test_images" + str(IMG_SIZE/2) + ".pickle", "wb")
    pickle.dump(test_images_half, pickle_out)
    pickle_out.close()

    pickle_out = open("test_images" + str(IMG_SIZE/4) + ".pickle", "wb")
    pickle.dump(test_images_quarter, pickle_out)
    pickle_out.close()

    pickle_out = open("test_labels.pickle", "wb")
    pickle.dump(test_labels, pickle_out)
    pickle_out.close()

create_testing_data(256)
