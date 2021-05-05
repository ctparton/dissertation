import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from pathlib import Path
from datetime import date
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import argparse
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=10, help="This is the batch size")
    parser.add_argument("--layers", default=0, help="This is the trainable layers")
    parser.add_argument("--lr", default=0.001, help="This is the learning rate")
    args = parser.parse_args()
    print(args)
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './regression_runsregression'
    BATCH_SIZE = int(args.b)
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'classification'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}/{TIME}_{MODEL_TYPE}_imdb_bede_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
    
    def img_to_raw_pixels(file_path):
        try:
            img = image.load_img(file_path, grayscale=False, target_size=(224, 224))
        except:
            return None
        x = preprocess_input(image.img_to_array(img))
        return x 
        
       

    if Path('imdb_wiki_data.pkl').is_file():
        print("Reading in data!")
        print("processed data exists")
        labels = pd.read_pickle('imdb_wiki_data.pkl')
    else:
        BASE_PATH = Path("../data/imdb_wiki_processed")
        labels = pd.read_csv(BASE_PATH / 'imdb_wiki_labels_t60.csv')
        labels = labels.astype({"image": 'string'})
        labels["imagepath"] = str(Path('../data/process_new/imdb_wiki_processed/loose_crop')) + "/" + labels['image']
        histogram_age = labels['age'].hist(bins=labels['age'].nunique())
        labels['image_pixels'] = labels['imagepath'].apply(img_to_raw_pixels)

    print("Done!")
    print(labels)
    labels = labels.dropna()

    target = labels['age'].values
    target_classes = keras.utils.to_categorical(target, 102)

    features = []
    for i in range(0, labels.shape[0]):
        features.append(labels['image_pixels'].values[i])

    features = np.array(features)
    # Reshape image pixels into a batch of 224 x 224 RGB images
    features = features.reshape(features.shape[0], 224, 224, 3)
    print(features.shape)
    features /= 255
    train_x, val_x, train_y, val_y = train_test_split(features, target_classes
                                        , test_size=0.20)
    
    train_gen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, zoom_range=[0.9,1.1])
    valid_gen = ImageDataGenerator()
    train_gen = train_gen.flow(x=train_x, y=train_y, batch_size=BATCH_SIZE)
    print(train_x.shape)
    print(train_y.shape)
    valid_gen = valid_gen.flow(x=val_x, y=val_y, batch_size=BATCH_SIZE)
    print(val_x.shape)
    print(val_y.shape)

    def age_mae(y_true, y_pred):
          true_age = K.sum(y_true * K.arange(0, 102, dtype="float32"), axis=-1)
          pred_age = K.sum(y_pred * K.arange(0, 102, dtype="float32"), axis=-1)
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
    model.add(Dropout(0.2))
    model.add(Dense(units=102, activation='softmax'))
    print(model.summary())

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='aug_final_loose_cropped_class_model.{epoch:02d}-{val_loss:.2f}.h5',
        save_weights_only=True,
        monitor='val_age_mae',
        mode='min',
        save_best_only=True)
    
    #model.compile(optimizer=Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])

    #model.fit(train_x, train_y
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
    model.compile(optimizer=Adam(LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])
    
  
    model.fit(x=train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
             epochs=30, verbose=2, callbacks=[model_checkpoint_callback])

    print("Weights saved")

