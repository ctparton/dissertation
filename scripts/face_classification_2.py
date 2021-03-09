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
# from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import coral_ordinal as coral


if __name__ == '__main__':
    print(tf.test.is_gpu_available())
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
    train["imagepath"] = BASE_PATH / 'train'
    train["imagepath"] = train["imagepath"].astype(str) + "/" + train['image']
    train = train.astype({"imagepath": 'string'})
    train = train.astype({"image": 'string'})
    train = train.dropna()
    val = pd.read_csv(BASE_PATH / 'valid_gt.csv')
    val = val.drop(columns=["stdv"])
    val["imagepath"] = BASE_PATH / 'valid'
    val["imagepath"] = val["imagepath"].astype(str) + "/" + val['image']
    val = val.astype({"imagepath": 'string'})
    val = val.astype({"image": 'string'})
    val = val.dropna()
    # print(train)
    # print(val)
    result = pd.concat([train, val])


    def img_to_raw_pixels(file_path):
        img = image.load_img(file_path, grayscale=False, target_size=(224, 224))
        x = preprocess_input(image.img_to_array(img))
        return x


    result['image_pixels'] = result['imagepath'].apply(img_to_raw_pixels)
    print(result)
    target = result['mean'].values
    target_classes = keras.utils.to_categorical(target, 90)

    features = []
    for i in range(0, result.shape[0]):
        features.append(result['image_pixels'].values[i])

    features = np.array(features)
    # Reshape image pixels into a batch of 224 x 224 RGB images
    features = features.reshape(features.shape[0], 224, 224, 3)
    print(features.shape)
    features /= 255
    train_x, val_x, train_y, val_y = train_test_split(features, target_classes
                                                      , test_size=0.20)
    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_gen = train_gen.flow(x=train_x, y=train_y, batch_size=BATCH_SIZE)
    print(train_x.shape)
    print(train_y.shape)
    valid_gen = valid_gen.flow(x=val_x, y=val_y, batch_size=BATCH_SIZE)
    print(val_x.shape)
    print(val_y.shape)


    def age_mae(y_true, y_pred):
        true_age = K.sum(y_true * K.arange(0, 90, dtype="float32"), axis=-1)
        pred_age = K.sum(y_pred * K.arange(0, 90, dtype="float32"), axis=-1)
        mae = K.mean(K.abs(true_age - pred_age))
        return mae


    vgg16_model = VGG16()
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
    model.add(coral.CoralOrdinal(num_classes = 90))
    print(model.summary())

    # model.compile(optimizer=Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])

    # model.fit(train_x, train_y
    #        , epochs=50
    #       , validation_data=(val_x, val_y),
    #       batch_size=BATCH_SIZE
    #      , verbose=2)

    # Set the last 8 layers to be trainable
    for layer in model.layers[14:]:
        layer.trainable = True
        print(f"{layer}: {layer.trainable}")
    print(model.summary())


    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=Adam(LR), loss=coral.OrdinalCrossEntropy(num_classes = 90), metrics=[coral.MeanAbsoluteErrorLabels()])

    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=30, verbose=2, callbacks=[tensorboard_callback])
