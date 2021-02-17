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
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from pathlib import Path
from datetime import date
from tensorflow.keras.preprocessing import image
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
    MODEL_TYPE = 'regression'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_bede_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

    # Preprocess data in csv
    BASE_PATH = Path("../data/regression")
    train = pd.read_csv(BASE_PATH / 'train_gt.csv')
    train = train.drop(columns=['stdv'])
    train["imagepath"] = train['image']
    print(train)
    train = train.astype({"imagepath": 'string'})
    train = train.astype({"image": 'string'})
    train = train.dropna()
    val = pd.read_csv(BASE_PATH / 'valid_gt.csv')
    val = val.drop(columns=["stdv"])
    val["imagepath"] = BASE_PATH / 'valid' / val['image']
    val = val.astype({"imagepath": 'string'})
    val = val.astype({"image": 'string'})
    val = val.dropna()

    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = train_gen.flow_from_dataframe(dataframe=train, x_col="imagepath", y_col="mean", class_mode="raw",
                                              target_size=(224, 224), batch_size=BATCH_SIZE)
    valid_gen = valid_gen.flow_from_dataframe(dataframe=val, x_col="imagepath", y_col="mean", class_mode="raw",
                                              target_size=(224, 224), batch_size=BATCH_SIZE)

    imgs, labels = next(train_gen)

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

    for layer in model.layers:
        layer.trainable = False

    # if not LAST_TRAINABLE_LAYERS:
    #     for layer in model.layers:
    #         layer.trainable = False
    # else:
    #     print(f"Setting trainable layers to {LAST_TRAINABLE_LAYERS}")
    #     for layer in vgg16_model.layers[:-LAST_TRAINABLE_LAYERS]:
    #         layer.trainable = False
    #         print(layer)

    model.add(Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=50, verbose=2)

    # Set the last 8 layers to be trainable
    for layer in model.layers[14:]:
        layer.trainable = True
        print(f"{layer}: {layer.trainable}")
    print(model.summary())

    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=30, verbose=2, callbacks=[tensorboard_callback])