import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from pathlib import Path
from datetime import date
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=10, help="This is the batch size")
    parser.add_argument("--layers", default=0, help="This is the trainable layers")
    parser.add_argument("--lr", default=0.001, help="This is the learning rate")
    args = parser.parse_args()
    print(args)
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './logs_bede/'
    BATCH_SIZE = int(args.b) 
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'classification'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_bede_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)


    classes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
               '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21,
               '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30, '31': 31,
               '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40, '41': 41,
               '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '50': 50, '51': 51,
               '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '60': 60, '61': 61,
               '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '70': 70, '71': 71,
               '72': 72, '73': 73,
               '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '80': 80, '81': 81, '82': 82, '83': 83}

    train_ds = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_directory(directory="../data/train", target_size=(224, 224), batch_size=BATCH_SIZE, classes=classes)
    valid_ds = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_directory(directory="../data/valid", target_size=(224, 224), batch_size=BATCH_SIZE, classes=classes)


    def age_mae(y_true, y_pred):
        true_age = K.sum(y_true * K.arange(0, 84, dtype="float32"), axis=-1)
        pred_age = K.sum(y_pred * K.arange(0, 84, dtype="float32"), axis=-1)
        mae = K.mean(K.abs(true_age - pred_age))
        return mae

    imgs, labels = next(train_ds)

    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    vgg16_model = VGG16()
    # vgg16_model.summary()
    model = keras.Sequential()
    # Take out last classification layer
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    # Train only last layer on first pass
    for layer in model.layers:
        layer.trainable = False
    # if not LAST_TRAINABLE_LAYERS:
    #     for layer in model.layers:
    #         layer.trainable = False
    # else:
    #     for layer in vgg16_model.layers[:-LAST_TRAINABLE_LAYERS]:
    #         layer.trainable = False

    model.add(Dense(units=84, activation='softmax'))
    model.compile(optimizer=Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])
    model.fit(x=train_ds, steps_per_epoch=len(train_ds), validation_data=valid_ds, validation_steps=len(valid_ds),
              epochs=30, verbose=2)
    # Set the last 8 layers to be trainable
    for layer in model.layers[14:]:
         layer.trainable = True
         print(f"{layer}: {layer.trainable}")
    print(model.summary())

    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=Adam(LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])
    model.fit(x=train_ds, steps_per_epoch=len(train_ds), validation_data=valid_ds, validation_steps=len(valid_ds),
              epochs=30, verbose=2, callbacks = [tensorboard_callback])
