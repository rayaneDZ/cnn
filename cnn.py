import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import cv2

import numpy as np
import pickle

train_input256 = pickle.load(open("train_images256.pickle", "rb")) #(1000, 256, 256, 3)

train_labels = pickle.load(open("train_labels256.pickle", "rb")) #(1000, 1)

# test_images = pickle.load(open("test_images256.pickle", "rb")) #(200, 256, 256, 3)

test_labels = pickle.load(open("test_labels256.pickle", "rb")) #(200, 1)

# print("train images shape: " + str(train_images.shape))
# print("train labels shape: " + str(train_labels.shape))
# print("test images shape: " + str(test_images.shape))
# print("test labels shape: " + str(test_labels.shape))

# class_names = ['dog', 'cat']

# model...
input64 = keras.Input(shape=(64, 64, 3), name="input64")
input128 = keras.Input(shape=(128, 128, 3), name="input128")
input256 = keras.Input(shape=(256, 256, 3), name="input256")

x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(input64)

y = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(input128)
y = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(y)
y = layers.MaxPooling2D(2)(y)

z = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(input256)
z = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(z)
z = layers.MaxPooling2D(2)(z)
z = layers.MaxPooling2D(2)(z)

w = layers.concatenate([x, y, z])
w = layers.Flatten()(w)
w = layers.Dense(10, activation='relu')(w)
w = layers.Dense(1, activation='softmax')(w)


model = keras.Model(inputs=[input64, input128, input256], outputs=w)
model.summary()
#plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs = 5

# fit
model.fit({"input64" : train_input64, "input128" : train_input128, "input256" : train_input256},
            train_labels256,
            epochs=epochs,
            batch_size=batch_size, verbose=2)

# evaulate
model.evaluate({"input64" : test_input64, "input128" : test_input128, "input256" : test_input256},
                test_labels256,
                batch_size=batch_size,
                verbose=2)