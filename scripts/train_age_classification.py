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
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tf_explain.core import OcclusionSensitivity


def get_param_args():
    parser = argparse.ArgumentParser(description='Pass hyperparameter arguments')
    parser.add_argument("--b", default=128, help="This is the batch size")
    parser.add_argument("--layers", default=0, help="This is the trainable layers")
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

def visualise_age_distribtion(dataset, mode):
    if mode == 'pretrain':
        dataset = dataset.astype({"age": 'int'})
        print(dataset)
        histogram_age = dataset['age'].hist(bins=dataset['age'].nunique())
    else:
        dataset = dataset.astype({"mean": 'int'})
        histogram_age = dataset['mean'].hist(bins=dataset['mean'].nunique())
        print(dataset)

    plt.xlabel("Age")
    plt.ylabel("Number of samples")
    # plt.show()
    # plt.savefig("chalearn.svg")


def load_data(mode):
    if mode == 'pretrain':
        if Path('imdb_wiki_data.pkl').is_file():
            print("Reading in data!")
            print("processed data exists")
            labels = pd.read_pickle('imdb_wiki_data.pkl')
        else:
            BASE_LABEL_PATH = Path("../data/imdb_wiki_processed")
            BASE_IMAGE_PATH = Path('../data/process_new/imdb_wiki_processed/loose_crop')
            result = pd.read_csv(BASE_PATH / 'imdb_wiki_labels_t60.csv')
            result = labels.astype({"image": 'string'})
            result["imagepath"] = str(BASE_IMAGE_PATH) + "/" + labels['image']
            result = result.dropna()
            return result
    else:
        BASE_LABEL_PATH = Path("../data/regression")
        BASE_IMAGE_PATH = Path("../data/process/aligned")

    # Preprocess data in csv
    train = pd.read_csv(BASE_LABEL_PATH / 'train_gt.csv')
    train = train.drop(columns=['stdv'])
    train["imagepath"] = BASE_IMAGE_PATH / 'aligned_train'
    train["imagepath"] = train["imagepath"].astype(str) + "/" + train['image']
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

def create_train_test_data(dataset, mode, test_ratio):
    dataset['image_pixels'] = dataset['imagepath'].apply(img_to_raw_pixels)
    # raw_dataset = raw_dataset.dropna()
    print(dataset)
    if mode == 'pretrain':
        target = dataset['age'].values
    else:
        target = dataset['mean'].values

    # One-Hot encode labels
    target_classes = keras.utils.to_categorical(target, 102)

    features = []
    # for each sample in the dataset
    for i in range(0, raw_dataset.shape[0]):
        # get the raw image pixels into the features array
        features.append(raw_dataset['image_pixels'].values[i])

    # Transform into NumPy representation
    features = np.array(features)
    # Reshape image pixels into batches of 224 x 224 RGB images
    features = features.reshape(features.shape[0], 224, 224, 3)
    print(features.shape)
    # apply additional rescaling
    features /= 255

    # produce a train/test data split using the test_ratio
    return train_test_split(features, target_classes, test_size=test_ratio)

def age_mae(y_true, y_pred):
    """
    Custom DEX Mean Absolute Error metric using the expected value formation
    (https://data.vision.ee.ethz.ch/cvl/publications/papers/proceedings/eth_biwi_01229.pdf)

    :param y_true: the softmax most probable class from prediction
    :param y_pred: the ground truth label
    :return: Mean Absolute Error metric for use as a custom Keras metric
    """
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
        vggface.add(Dense(units=102, activation='softmax'))
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
        filepath='aug_final_loose_cropped_class_model.{epoch:02d}-{val_loss:.2f}.h5',
        save_weights_only=True,
        monitor='val_age_mae',
        mode='min',
        save_best_only=True)

if __name__ == '__main__':
    args = get_param_args()
    tf.config.list_physical_devices('GPU')
    BASE_LOG_PATH = './aligned_logs/'
    BATCH_SIZE = int(args.b)
    LAST_TRAINABLE_LAYERS = int(args.layers)
    LR = float(args.lr)
    MODEL_TYPE = 'classification'
    TIME = str(date.today()).replace(" ", "-")
    LOG_DIR = f"{BASE_LOG_PATH}{MODEL_TYPE}/{TIME}_final_aligned_loose_aw_bede_batch_{BATCH_SIZE}_lr_{LR}_layers_{LAST_TRAINABLE_LAYERS}"
    print(f"Logging to {LOG_DIR}")
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

    raw_dataset = load_data(args.mode)
    print(raw_dataset)
    visualise_age_distribtion(raw_dataset, args.mode)

    train_x, val_x, train_y, val_y = create_train_test_data(raw_dataset, args.mode, 0.20)

    # augmentation of the training data
    train_gen = ImageDataGenerator(horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=[0.9, 1.1],
                                   rotation_range=10)

    valid_gen = ImageDataGenerator()
    train_gen = train_gen.flow(x=train_x,
                               y=train_y,
                               batch_size=BATCH_SIZE,
                               save_to_dir="augmented",
                               save_prefix="aug_",
                               )
    valid_gen = valid_gen.flow(x=val_x, y=val_y, batch_size=BATCH_SIZE)

    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)

    model = construct_model(args.type)

    print(model.summary())
    # Recompile model with new learning rate and last 8 layers trainable
    model.compile(optimizer=Adam(LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', age_mae])

    callbacks = [tensorboard_callback]

    if args.mode == 'pretrain':
        model_checkpoint_callback = create_checkpoint_callback()
        callbacks.append(model_checkpoint_callback)

    model.fit(x=train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
              epochs=30, verbose=2)
    # , callbacks=[tensorboard_callback]
    # model.save(Path("final_models/class_model"))

