import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import numpy as np
import pickle

train_images = pickle.load(open("train_images.pickle", "rb")) #(498, 72, 72, 3)

train_labels = pickle.load(open("train_labels.pickle", "rb")) #(498, 1)

test_images = pickle.load(open("test_images.pickle", "rb")) #(100, 72, 72, 3)

test_labels = pickle.load(open("test_labels.pickle", "rb")) #(100, 1)

import sys
sys.exit()

class_names = ['dog', 'cat']

# model...
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(train_images, train_labels, epochs=epochs,
          batch_size=batch_size, verbose=2)

# evaulate
model.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)