from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path
from tensorflow.keras import backend as K
import keract
import numpy as np
from tf_explain.core import OcclusionSensitivity


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 102, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 102, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_saved_model(model_name):
    model = keras.models.load_model(f'final_models/{model_name}', custom_objects={'age_mae': age_mae})
    return model


def prepare_image(img_path, mode="activation"):
    pil_image = keras.preprocessing.image.load_img(str(img_path),
                                                   grayscale=False, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(pil_image)
    if mode == 'activation':
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    img_array = preprocess_input(img_array)
    img_array /= 255
    return img_array


def predict_on_image(image, model):
    yhat = model.predict(image)
    print("Dominant Class: ", np.argmax(yhat))
    apparent_age = np.round(K.sum(yhat * K.arange(0, 102, dtype="float32"), axis=-1))
    print("Apparent Age: ", int(apparent_age[0]))


def get_layer_outputs(image_path, model):
    image = prepare_image(image_path)
    print(predict_on_image(image, model))
    activations = keract.get_activations(model, image)
    keract.display_heatmaps(activations, image, directory=f"../visualisations/activation_new/{image_path.name}",
                            save=True)


def visualise_occlusion_sensitivity(image_path, model, target_age):
    SAVE_PATH = f'../visualisations/occlusion/{image_path.stem}'
    explainer = OcclusionSensitivity()
    image = prepare_image(image_path, mode="occlusion")
    print(image.shape)
    print(predict_on_image(prepare_image(image_path), model))

    for i in range(3, 30):
        grid = explainer.explain(([image], None), model, target_age, patch_size=i, colormap=2)
        explainer.save(grid, SAVE_PATH, f'patch_{i}_age_{target_age}.png')

        grid = explainer.explain(([image], None), model, target_age, patch_size=i)
        explainer.save(grid, SAVE_PATH, f'default_colour_patch_{i}_age_{target_age}.png')


if __name__ == "__main__":
    BASE_PATH = Path("C:\\Uni\\DISS\\project\\data\\process\\aligned\\aligned_test")
    model = get_saved_model("class_model_nodrop_early_512")
    visualise_occlusion_sensitivity(Path(BASE_PATH / Path("005651.jpg")), model, 25)
    visualise_occlusion_sensitivity(Path(BASE_PATH / Path("005913.jpg")), model, 19)
    visualise_occlusion_sensitivity(Path(BASE_PATH / Path("005681.jpg")), model, 44)
