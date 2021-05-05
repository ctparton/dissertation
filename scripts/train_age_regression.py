import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from pathlib import Path
from datetime import date
from tensorflow.keras.preprocessing import image
import argparse

def get_param_args():
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=64, help="This is the batch size")
    parser.add_argument("--layers", default=8, help="This is the trainable layers")
    parser.add_argument("--lr", default=0.0001, help="This is the learning rate")
    parser.add_argument("--type", default='VGG16', help="This is the model architecture (VGG16 or VGGFACE)")
    parser.add_argument("--mode", default="finetune", help="Pass 'finetune' for ChaLearn training or 'pretrain for IMDB-wiki")
    args = parser.parse_args()
    if is_valid_arch(args.type) and is_valid_training_strategy(args.mode):
        return args
    else:
        return None

def is_valid_arch(arch_type):
    """
    Checks if model architecture is one of the two architectures
    :param model_type: A (string) type of architecture
    :return: True if arch_type is valid, else raises an error
    """
    valid_types = {'VGG16', 'VGGFACE'}
    if arch_type not in valid_types:
        raise ValueError("'type' must be one of %r." % valid_types)
    return True

def is_valid_training_strategy(training_mode):
    """
    Checks if either IMDB-Wiki (pre-train) or ChaLearn is selected (finetune_
    :param model_type: A (string) type of training
    :return: True is training_mode is valid else raises an error
    """
    valid_types = {'finetune', 'pretrain'}
    if training_mode not in valid_types:
        raise ValueError("'mode' must be one of %r." % valid_types)
    return True

def plotImages(images_arr):
   fig, axes = plt.subplots(1, 10, figsize=(20, 20))
   axes = axes.flatten()
   for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
   plt.tight_layout()
   plt.show()

def load_data(mode):
    if mode == 'pretrain':
        if Path('imdb_wiki_data.pkl').is_file():
            print("Reading in data!")
            print("processed data exists")
            labels = pd.read_pickle('imdb_wiki_data.pkl')
        else:
            BASE_LABEL_PATH = Path("../data/imdb_wiki_processed")
            BASE_IMAGE_PATH = Path('../data/process_new/imdb_wiki_processed/loose_crop')
            result = pd.read_csv(BASE_PATH / 'imdb_wiki_labels.csv')
            result = labels.astype({"image": 'string'})
            result['age'] = result['age'].astype(float)
            result["imagepath"] = str(BASE_IMAGE_PATH) + "/" + labels['image']
            result = result.dropna()
            train = result[result['partition'] == 'train']
            val = result[result['partition'] == 'valid']
            return (train, val)
    else:
        BASE_LABEL_PATH = Path("../data/regression")
        BASE_IMAGE_PATH = Path("../data/process/aligned")

    # Preprocess data in csv
    train = pd.read_csv(BASE_LABEL_PATH / 'train_gt.csv')
    train = train.drop(columns=['stdv'])
    train["imagepath"] = BASE_IMAGE_PATH / 'aligned_train'
    train["imagepath"] = train["imagepath"].astype(str) + "/" + train['image']
    print(train)
    train = train.astype({"imagepath": 'string'})
    train = train.astype({"image": 'string'})
    train = train.dropna()
    val = pd.read_csv(BASE_LABEL_PATH / 'valid_gt.csv')
    val = val.drop(columns=["stdv"])
    val["imagepath"] = BASE_IMAGE_PATH / 'aligned_valid'
    val["imagepath"] = val["imagepath"].astype(str) + "/" + val['image']
    val = val.astype({"imagepath": 'string'})
    val = val.astype({"image": 'string'})
    val = val.dropna()
    test = pd.read_csv(Path('../data/') / 'test_gt.csv')
    test = test.drop(columns=['stdv'])
    test["imagepath"] = BASE_IMAGE_PATH / 'aligned_test'
    test["imagepath"] = test["imagepath"].astype(str) + "/" + test['image']
    test = test.astype({"imagepath": 'string'})
    test = test.astype({"image": 'string'})
    test = test.dropna()
    return (train, val, test)

def construct_model(type):
    if type == 'VGG16':
        vgg16_model = VGG16(weights='aug_loose_cropped_model.17-6.47.h5', classes=1)
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
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Set the last 8 layers to be trainable
        for layer in model.layers[14:]:
            layer.trainable = True
            print(f"{layer}: {layer.trainable}")
        return model
    elif type == 'VGGFACE':
        # https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
        vggface = tf.keras.models.Sequential()
        vggface.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        vggface.add(Convolution2D(64, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(64, (3, 3), activation='relu'))
        vggface.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(128, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(128, (3, 3), activation='relu'))
        vggface.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(256, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(256, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(256, (3, 3), activation='relu'))
        vggface.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(ZeroPadding2D((1, 1)))
        vggface.add(Convolution2D(512, (3, 3), activation='relu'))
        vggface.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggface.add(Convolution2D(4096, (7, 7), activation='relu'))
        vggface.add(Dropout(0.5))
        vggface.add(Convolution2D(4096, (1, 1), activation='relu'))
        vggface.add(Dropout(0.5))
        vggface.add(Convolution2D(2622, (1, 1)))
        vggface.add(Flatten())
        vggface.add(Activation('softmax'))

        vggface.load_weights('./face_weights/vgg_face_weights.h5')

        vggface.pop()
        vggface.pop()
        vggface.pop()
        vggface.add(Flatten())
        vggface.add(Dense(units=1))
        for layer in vvgface.layers:
            layer.trainable = False
        # Set the last 8 layers to be trainable
        for layer in vvgface.layers[14:]:
            layer.trainable = True
            print(f"{layer}: {layer.trainable}")
        print(vvgface.summary())
        return vggface
    else:
        return None

def create_checkpoint_callback():
    return tf.keras.callbacks.ModelCheckpoint(
    filepath='aug_loose_cropped_model.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

def evaluate_on_test(model):
    results = model.evaluate(test_gen, steps=len(test_gen))
    results = {out: results[i] for i, out in enumerate(model.metrics_names)}
    return results

if __name__ == '__main__':
    args = get_param_args()
    print(args)
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './aligned_logs/'
    BATCH_SIZE = int(args.b)
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'regression'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}__finalfinal_aligned_loose_aw_batch_{BATCH_SIZE}_lr_{LR}_2p_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

    if args.mode == 'pretrain':
        train, val = load_data(args.mode)
        y_col = 'age'
    else:
        train, val, test = load_data(args.mode)
        test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_gen = test_gen.flow_from_dataframe(dataframe=val, x_col="imagepath", y_col="mean", class_mode="raw",
                                                target_size=(224, 224), batch_size=BATCH_SIZE, shuffle=False)
        y_col = 'mean'

    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, zoom_range=[0.9,1.1], rotation_range=10)
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_gen.flow_from_dataframe(dataframe=train, x_col="imagepath", y_col=y_col, class_mode="raw",
                                              target_size=(224, 224), batch_size=BATCH_SIZE)
    valid_gen = valid_gen.flow_from_dataframe(dataframe=val, x_col="imagepath", y_col=y_col, class_mode="raw",
                                              target_size=(224, 224), batch_size=BATCH_SIZE)

    # Plot training samples as a check
    # imgs, labels = next(train_gen)

    model = construct_model(args.type)

    callbacks = [tensorboard_callback]

    if args.mode == 'pretrain':
        model_checkpoint_callback = create_checkpoint_callback()
        callbacks.append(model_checkpoint_callback)
    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    print(model.summary())
    model.fit(x=train_gen, steps_per_epoch=len(train_gen), validation_data=valid_gen, validation_steps=len(valid_gen),
              epochs=50, verbose=2)
    # callbacks=[tensorboard_callback]
    print(f"test results {evaluate_on_test(model)}")
    # model.save(Path("final_models/regression_model"))



