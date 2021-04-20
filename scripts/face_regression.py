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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=64, help="This is the batch size")
    parser.add_argument("--layers", default=8, help="This is the trainable layers")
    parser.add_argument("--lr", default=0.0001, help="This is the learning rate")
    args = parser.parse_args()
    print(args)
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './cropped_logs/'
    BATCH_SIZE = int(args.b)
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'regression'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_bede_aw_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

    # Preprocess data in csv
    BASE_PATH = Path("../data/regression")
    CROPPED_BASE_PATH = Path("../data/process/regression")
    train = pd.read_csv(BASE_PATH / 'train_gt.csv')
    train = train.drop(columns=['stdv'])
    train["imagepath"] = CROPPED_BASE_PATH / 'train'
    train["imagepath"] =  train["imagepath"].astype(str) + "/" + train['image']
    print(train)
    train = train.astype({"imagepath": 'string'})
    train = train.astype({"image": 'string'})
    train = train.dropna()
    val = pd.read_csv(BASE_PATH / 'valid_gt.csv')
    val = val.drop(columns=["stdv"])
    val["imagepath"] = CROPPED_BASE_PATH / 'valid'
    val["imagepath"] = val["imagepath"].astype(str) + "/" + val['image']
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

    vgg16_model = VGG16(weights='cropped_model.06-6.74.h5', classes=1)
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
    model.add(Dropout(0.4))
    model.add(Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    print(model.summary())
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=50, verbose=2)

    # Set the last 8 layers to be trainable
    for layer in model.layers[14:]:
        layer.trainable = True
        print(f"{layer}: {layer.trainable}")

    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    print(model.summary())
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=50, verbose=2, callbacks=[tensorboard_callback])


    img_array = preprocess_input(keras.preprocessing.image.img_to_array(image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\000013.jpg')),
                           grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("000013", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004124.jpg')),
                           grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("004124: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004138.jpg')),
                           grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("004138: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004385.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("004385: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\test2.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("dad 1.jpg: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004138.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("004138: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\mum.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("mum.jpg: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\dad.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("dad.jpg: ", yhat)

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\amy.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    yhat = model.predict(img_array)
    print("amy.jpg: ", yhat)
