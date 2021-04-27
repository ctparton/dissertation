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
from tf_explain.core import OcclusionSensitivity


def get_param_args():
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=128, help="This is the batch size")
    parser.add_argument("--layers", default=0, help="This is the trainable layers")
    parser.add_argument("--lr", default=0.0001, help="This is the learning rate")
    parser.add_argument("--type", default='VGG16', help="This is the model architecture (VGG16 or VGGFACE)")
    args = parser.parse_args()
    return args


def load_data():
    # Preprocess data in csv
    BASE_PATH = Path("../data/regression")
    CROPPED_BASE_PATH = Path("../data/process/aligned")
    train = pd.read_csv(BASE_PATH / 'train_gt.csv')
    train = train.drop(columns=['stdv'])
    train["imagepath"] = CROPPED_BASE_PATH / 'aligned_train'
    train["imagepath"] = train["imagepath"].astype(str) + "/" + train['image']
    train = train.astype({"imagepath": 'string'})
    train = train.astype({"image": 'string'})
    train = train.dropna()
    val = pd.read_csv(BASE_PATH / 'valid_gt.csv')
    val = val.drop(columns=["stdv"])
    val["imagepath"] = CROPPED_BASE_PATH / 'aligned_valid'
    val["imagepath"] = val["imagepath"].astype(str) + "/" + val['image']
    val = val.astype({"imagepath": 'string'})
    val = val.astype({"image": 'string'})
    val = val.dropna()
    result = pd.concat([train, val])
    return result

def img_to_raw_pixels(file_path):
    try:
        img = image.load_img(file_path, grayscale=False, target_size=(224, 224))
    except:
        return None
    x = preprocess_input(image.img_to_array(img))
    x = image.img_to_array(img)
    return x

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 102, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 102, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def construct_model(type):
    if type == 'VGG16':
        vgg16_model = VGG16(weights='aug_loose_cropped_class_model.07-3.53.h5', classes=102)
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
        model.add(Dropout(0.2))
        model.add(Dense(units=102, activation='softmax'))
        # Set the last 8 layers to be trainable
        for layer in model.layers[14:]:
            layer.trainable = True
            print(f"{layer}: {layer.trainable}")
        print(model.summary())
        return model
    elif type == 'VGGFACE':
        return None
    else:
        return None


if __name__ == '__main__':
    args = get_param_args()
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './aligned_logs/'
    BATCH_SIZE = int(args.b)
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'classification'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_ffinalaug_aligned_loose3_aw_bede_batch_{BATCH_SIZE}_lr_{LR}_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

    raw_dataset = load_data()
    print(raw_dataset)

    raw_dataset['image_pixels'] = raw_dataset['imagepath'].apply(img_to_raw_pixels)
    # raw_dataset = raw_dataset.dropna()
    print(raw_dataset)
    target = raw_dataset['mean'].values
    target_classes = keras.utils.to_categorical(target, 102)

    features = []
    # for each sample in the dataset
    for i in range(0, raw_dataset.shape[0]):
        features.append(raw_dataset['image_pixels'].values[i])

    features = np.array(features)
    # Reshape image pixels into a batch of 224 x 224 RGB images
    features = features.reshape(features.shape[0], 224, 224, 3)
    print(features.shape)
    # apply additional rescaling
    features /= 255
    train_x, val_x, train_y, val_y = train_test_split(features, target_classes
                                                      , test_size=0.20)

    # augmentation of the training data
    train_gen = ImageDataGenerator(horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=[0.9,1.1],
                                   rotation_range=10)

    valid_gen = ImageDataGenerator()
    train_gen = train_gen.flow(x=train_x,
                               y=train_y,
                               batch_size=BATCH_SIZE,
                               save_to_dir="augmented",
                               save_prefix="aug_",
                              )
    print(train_x.shape)
    print(train_y.shape)
    valid_gen = valid_gen.flow(x=val_x, y=val_y, batch_size=BATCH_SIZE)
    print(val_x.shape)
    print(val_y.shape)

    model = construct_model(args.type)

    print(model.summary())
    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=Adam(LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])

    model.fit(x=train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
             epochs=30, verbose=2)
    #, callbacks=[tensorboard_callback]
    model.save(Path("final_models/class_model"))
    explainer = OcclusionSensitivity() #41
    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\process\\regression\\train\\000102.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("000013", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    BASE_VIS_PATH = Path('C:\\Uni\\DISS\\project\\data\\process\\aligned\\aligned_test')
    img_inp = tf.keras.preprocessing.image.load_img(
        str(Path(BASE_VIS_PATH / Path("005616.jpg"))), target_size=(224, 224))
    img_inp = preprocess_input(tf.keras.preprocessing.image.img_to_array(img_inp))
    img_inp /= 255
    grid = explainer.explain(([img_inp], None), model, 24, patch_size=30, colormap=2)
    explainer.save(grid, '.', 'occlusion_sensitivity_6.png')
    grid = explainer.explain(([img_inp], None), model, 24, patch_size=20, colormap=2)
    explainer.save(grid, '.', 'occlusion_sensitivity_7.png')
    grid = explainer.explain(([img_inp], None), model, 24, patch_size=5, colormap=2)
    explainer.save(grid, '.', 'occlusion_sensitivity_8.png')

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path(BASE_VIS_PATH / Path("005616.jpg"))),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("test 005616: ", yhat)

    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004138.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("004138: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004385.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("004385: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\test2.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("dad 1.jpg: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\004138.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("004138: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\mum.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("mum.jpg: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\dad.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("dad.jpg: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))

    img_array = preprocess_input(keras.preprocessing.image.img_to_array(
        image.load_img(str(Path('C:\\Uni\\DISS\\project\\data\\test\\amy.jpg')),
                       grayscale=False, target_size=(224, 224))))
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255
    yhat = np.argmax(model.predict(img_array), axis=-1)
    print("amy.jpg: ", yhat)
    apparent_age = np.round(K.sum(model.predict(img_array) * K.arange(0, 102, dtype="float32"), axis=-1))
    print("apparent age: ", int(apparent_age[0]))