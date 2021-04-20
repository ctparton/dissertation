from io import BytesIO

import requests
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from pathlib import Path
from datetime import date
from tensorflow.keras.preprocessing import image
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import keract
import seaborn as sns
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import pylab
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from numpy.random import permutation
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":
    # Check for GPUs and set them to dynamically grow memory as needed
    # Avoids OOM from tensorflow greedily allocating GPU memory
    vgg16_model = VGG16(weights='final_model.03-3.55.h5', classes=102)
    vgg16_model.summary()
    model = keras.Sequential()
    # Take out last classification layer
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    # Train only last layer on first pass
    for layer in model.layers:
        layer.trainable = False
    # # if not LAST_TRAINABLE_LAYERS:
    # #     for layer in model.layers:
    # #         layer.trainable = False
    # # else:
    # #     for layer in vgg16_model.layers[:-LAST_TRAINABLE_LAYERS]:
    # #         layer.trainable = False
    #
    model.add(Dropout(0.4))
    model.add(Dense(units=102, activation='softmax'))
    print(model.summary())
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    image = image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\000013.jpg')),
                           grayscale=False, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("most dominant age class (not apparent age): ", np.argmax(yhat))
    apparent_age = np.round(K.sum(yhat * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    activations = keract.get_activations(model, image)
    first = activations.get('block1_conv1')
    keract.display_heatmaps(activations, image, save=True)
    keract.display_activations(activations, save=True)

