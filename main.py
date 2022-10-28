import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Input

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0

# Sequential API (convenient, but not flexible) 1 input mapped to 1 output
model = Sequential(
    [
        layers.InputLayer(input_shape=28*28),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)

    ]
)

print(model.summary())

model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]

)

model.fit(x_train,y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)