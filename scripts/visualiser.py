from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path
from tensorflow.keras import backend as K
import keract
import numpy as np
from tf_explain.core import OcclusionSensitivity
from tensorflow.keras.utils import plot_model
import pylab
import seaborn as sns
import math
import matplotlib.pyplot as plt
import cv2
import copy

# Code adapted from https://github.com/oswaldoludwig/Sensitivity-to- occlusion - Keras -
def visualise_occlusion_sensitivity(image_path, model, occluding_size=70, override_class=None, save_dir='../visualisations/patches/'):
    '''
    Produces a heatmap visualisation of sensitivity to occlusion

    :param image_path: file path of the raw image to pass to the model
    :param model: Keras SavedModel instance
    :param occluding_size: size of the grey occlusion patch
    :param override_class: (optional) to produce the heatmap for a particular class
    :param save_dir: (optional) directory to save result in
    :return: void
    '''
    occluding_stride = 5

    save_path = f"{save_dir}{image_path.stem}"
    image = cv2.imread(str(image_path))
    im = prepare_image_open_cv(image)
    out = model.predict(im)
    out = out[0]

    if override_class is not None:
        print("modifying")
        target_class = override_class
    else:
        target_class = np.argmax(out)

    height, width, _ = image.shape
    output_height = int(math.ceil((height - occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width - occluding_size) / occluding_stride + 1))
    # init heatmap
    heatmap = np.zeros((output_height, output_width))

    for h_pixel in range(output_height):
        for w_pixel in range(output_width):
            # Size of occlusion patch:
            h_start = h_pixel * occluding_stride
            h_end = min(height, h_start + occluding_size)

            w_start = w_pixel * occluding_stride
            w_end = min(width, w_start + occluding_size)

            # Getting the image copy, applying the occluding window and predicting to obtain prob:
            input_image = copy.copy(image)
            input_image[h_start:h_end, w_start:w_end, :] = 128
            im = prepare_image_open_cv(input_image)
            out = model.predict(im)
            # compare to prob without occlusion
            prob = out[0][target_class]
            # invert to highlight important areas in warmer tones
            heatmap[h_pixel, w_pixel] = 1 - prob

    f = pylab.figure()
    #https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ax = sns.heatmap(heatmap, cmap='jet', xticklabels=False, yticklabels=False)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    if override_class is not None:
        override = "yes_ov"
    else:
        override = "no_ov"

    plt.savefig(f"{save_path}/{occluding_size}_{override}_pos3.png")
    print('Predicted %s' % target_class)


def prepare_image_open_cv(cv_image):
    '''
    Alt method to prepare image with opencv
    :param cv_image: image read using opencv imread
    :return: processed image
    '''
    im = cv2.resize(cv_image, (224, 224)).astype(np.float32)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    im /= 255
    return im


def is_valid_model_type(model_type):
    """
    Checks if model is one of the two implemented types
    :param model_type: A (string) type of model, either classification or regression
    :return: True is model_type is valid else raises an error
    """
    valid_types = {'classification', 'regression'}
    if model_type not in valid_types:
        raise ValueError("'model_type' must be one of %r." % valid_types)
    return True


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


def get_saved_model(model_name, model_type, create_plot=False):
    """
    Retrieves the saved model in Keras SavedModel format

    :param model_name: Root directory of the SavedModel
    :param model_type: A (string) type of model, either classification or regression
    :param create_plot: (optional) create a plot of the model
    :return: The SavedModel
    """
    if not is_valid_model_type(model_type):
        raise RuntimeError("Model type could not be verified")

    if model_type == 'classification':
        model = keras.models.load_model(
            f"final_models/{model_name}", custom_objects={"age_mae": age_mae}
        )
    else:
        model = keras.models.load_model(
            f"final_models/{model_name}"
        )
    if create_plot:
        plot_model(model, to_file='model.png', rankdir='LR', show_shapes=True, show_layer_names=True)
    return model


def prepare_image(img_path, model_type, mode="activation"):
    '''
    Prepares raw images to be passed to the model at inference time

    :param img_path: file path to the image
    :param model_type: A type of model, either classification or regression
    :param mode: Default: 'activation', preparation type for visualisation technique
    :return: NumPy array representation of the processed image
    '''
    if not is_valid_model_type(model_type):
        raise RuntimeError("Model type could not be verified")

    pil_image = keras.preprocessing.image.load_img(
        str(img_path), grayscale=False, target_size=(224, 224)
    )
    img_array = keras.preprocessing.image.img_to_array(pil_image)
    if mode == "activation":
        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
        )
    img_array = preprocess_input(img_array)
    if model_type == 'classification':
        img_array /= 255
    return img_array


def predict_on_image(image, model, model_type):
    """
    Runs the model in inference to obtain a prediction on a processed image

    In classification mode this will output the most probable class and also the DEX expected value formation
    (https://data.vision.ee.ethz.ch/cvl/publications/papers/proceedings/eth_biwi_01229.pdf)

    :param image: processed image to pass to the model
    :param model: Keras Model instance in SavedModel format
    :param model_type: A type of model, either classification or regression
    :return: void
    """
    yhat = model.predict(image)
    print("Highest Probability Class: ", np.argmax(yhat))
    if model_type == 'classification':
        perceived_age = np.round(K.sum(yhat * K.arange(0, 102, dtype="float32"), axis=-1))
        print("Perceived Age (Expected Value): ", int(perceived_age[0]))


def prepare_predict(image_path, model, model_type):
    """
    Runs the model in inference to obtain a prediction on a RAW input image after applying processing

    :param image_path: file path of the image to pass to the model
    :param model: Keras Model instance in SavedModel format
    :param model_type: A type of model, either classification or regression
    :return: void
    """
    image = prepare_image(image_path, model_type)
    print(predict_on_image(image, model, model_type))


def visualise_layer_outputs(image_path, model, model_type, save_dir='../visualisations/activation_test/'):
    """
    Retrieves the feature maps for all layers in the VGG16 model, with heatmaps overlaid

    :param image_path: file path of the image to pass to the model
    :param model: Keras Model instance in SavedModel format
    :param model_type: A type of model, either classification or regression
    :param save_dir: (optional) directory to save visualisations in
    :return: void
    """
    save_path = f"{save_dir}{image_path.stem}"
    image = prepare_image(image_path, model_type)
    print(predict_on_image(image, model, model_type))
    activations = keract.get_activations(model, image)
    keract.display_heatmaps(
        activations,
        image,
        directory=save_path,
        save=True,
    )

# OLD implementation
def visualise_occlusion_sensitivity_old(image_path, model, target_age, model_type, save_dir='../visualisations/occlusion_test/'):
    """
    Creates an occlusion sensitivity visualisation with varying patch sizes for the given model

    :param image_path: file path of the image to pass to the model
    :param model: Keras Model instance in SavedModel format
    :param target_age: Target age for computing heatmaps during occlusion process
    :param model_type: A type of model, either classification or regression
    :param save_dir: (optional) directory to save visualisations in
    :return: void
    """
    save_path = f"{save_dir}{image_path.stem}"
    explainer = OcclusionSensitivity()
    image = prepare_image(image_path, model_type, mode="occlusion")
    print(image.shape)
    print(predict_on_image(prepare_image(image_path, model_type), model, model_type))

    for i in range(3, 30):
        grid = explainer.explain(
            ([image], None), model, target_age, patch_size=i, colormap=2
        )
        explainer.save(grid, save_path, f"patch_{i}_age_{target_age}.png")

        grid = explainer.explain(([image], None), model, target_age, patch_size=i)
        explainer.save(
            grid, save_path, f"default_colour_patch_{i}_age_{target_age}.png"
        )


if __name__ == "__main__":
    # Path to ChaLearn V2 test images
    BASE_PATH = Path("C:\\Uni\\DISS\\project\\data\\process\\aligned\\aligned_test")
    MODEL_TYPE = 'classification'
    # Visualisations were performed using the classifiction model for compatibility reasons, this model performed
    # similarly to the best performing model
    saved_model = get_saved_model("class_model_nodrop_early_512", MODEL_TYPE)
    print("005681")
    prepare_predict(Path(BASE_PATH / Path("005681.jpg")), saved_model, MODEL_TYPE)
    visualise_occlusion_sensitivity(Path(BASE_PATH / Path("005681.jpg")), saved_model, occluding_size=30)
    # print("005628")
    # prepare_predict(Path(BASE_PATH / Path("005628.jpg")), saved_model, MODEL_TYPE)
    # print("005625")
    # prepare_predict(Path(BASE_PATH / Path("005625.jpg")), saved_model, MODEL_TYPE)
    # print("005681")
    # prepare_predict(Path(BASE_PATH / Path("005681.jpg")), saved_model, MODEL_TYPE)
    # print("005651")
    # prepare_predict(Path(BASE_PATH / Path("005651.jpg")), saved_model, MODEL_TYPE)
