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
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=10, help="This is the batch size")
    parser.add_argument("--layers", default=0, help="This is the trainable layers")
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
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_imdb_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"Time {current_time}")
    # Preprocess data in csv
    BASE_PATH = Path("../data/imdb_wiki_processed")
    labels = pd.read_csv(BASE_PATH / 'imdb_wiki_labels.csv')
    labels = labels.astype({"image": 'string'})
    labels['age'] = labels['age'].astype(float)
    labels["imagepath"] = str(Path("../data/process_new/imdb_wiki_processed/loose_crop")) + "/" + labels['image']
    labels = labels.astype({"imagepath": 'string'})
    train = labels[labels['partition'] == 'train']
    valid = labels[labels['partition'] == 'valid']

    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, zoom_range=[0.9,1.1], rotation_range=10)
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = train_gen.flow_from_dataframe(dataframe=train, x_col="imagepath", y_col="age", class_mode="raw",
                                              target_size=(224, 224), batch_size=BATCH_SIZE)
    valid_gen = valid_gen.flow_from_dataframe(dataframe=valid, x_col="imagepath", y_col="age", class_mode="raw",
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
    model.add(Dropout(0.4))
    model.add(Dense(units=1))

    #model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    #model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
     #         epochs=50, verbose=2)

    # Set the last 8 layers to be trainable
    for layer in model.layers[14:]:
        layer.trainable = True
        print(f"{layer}: {layer.trainable}")
    print(model.summary())
       

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='aug_loose_cropped_model.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=30, verbose=2, callbacks=[tensorboard_callback, model_checkpoint_callback])
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    model.save_weights(f'imdb_wiki_weights_{current_time}.h5')
    print("Weights saved")
