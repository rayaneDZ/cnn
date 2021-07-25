import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import pickle

DATADIR = "../Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 72

train_images = []
train_labels = []

def create_training_data(train_images, train_labels, IMG_SIZE):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path)[:250]:
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255
                plt.imshow(new_array)
                plt.show()
            except Exception as e:
                print("an image failed")
            break
        break

create_training_data(train_images, train_labels, 72)
create_training_data(train_images, train_labels, 128)
create_training_data(train_images, train_labels, 256)
