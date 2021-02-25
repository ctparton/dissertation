import face_recognition
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm import tqdm


def dlib_face_detector(images, retry):
    error_list = []
    image_back = images
    print(images)
    resized_faces = [cv2.resize(cv2.imread(str(image), cv2.IMREAD_UNCHANGED)[..., ::-1], dsize=(299, 299)) for image in
                     images]
    face_locations = face_recognition.batch_face_locations(resized_faces)
    for i in range(len(resized_faces)):
        print(f"resized_faces {len(resized_faces)}")
        print(f"locations {len(face_locations)}")
        print(f"images {len(image_back)}")
        if not face_locations[i]:
            error_list.append(images[i])
        else:
            print(face_locations[i][0])
            top, right, bottom, left = face_locations[i][0]
            image = resized_faces[i]
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)

            if retry:
                print("running in retry mode")
                process_train_path = Path(PROCESS_PATH, images[i].parent.parent.parent.name,
                                          images[i].parent.parent.name)
                pil_image.save(Path(process_train_path, Path(images[i].name)))
                print(f"removing image {images[i]}")
                os.remove(images[i])
            else:
                Path(PROCESS_PATH, Path(images[i].parent.name)).mkdir(exist_ok=True)
                pil_image.save(Path(PROCESS_PATH, Path(images[i].parent.name), Path(images[i].name)))
    return error_list


def single_dlib_face_detector(img, mode):
    print(img.name)
    error_list = []
    image = face_recognition.load_image_file(img)
    try:
        face_locations = face_recognition.face_locations(image, model="cnn")
        print(face_locations)
        if not face_locations:
            error_list.append(img)
            Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry").mkdir(exist_ok=True, parents=True)
            process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry")
            pil_image.save(Path(process_train_path, Path(img.name)))
        else:
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            if 'class' in mode:
                Path(PROCESS_PATH, img.parent.parent.name, img.parent.name).mkdir(exist_ok=True, parents=True)
                process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name)
                pil_image.save(Path(process_train_path, Path(img.name)))
            else:
                Path(PROCESS_PATH, img.parent.parent.name, img.parent.name).mkdir(exist_ok=True, parents=True)
                process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name)
                pil_image.save(Path(process_train_path, Path(img.name)))

    except Exception as e:
        error_list.append(img)
        Path(img.parent, Path("exclude")).mkdir(exist_ok=True)
        print(f"error {str(e)}")
        pil_image = Image.fromarray(face_recognition.load_image_file(img))
        print("ERR COPYING FILE")
        if 'class' in mode:
            Path(PROCESS_PATH, img.parent.parent.name, img.parent.name).mkdir(exist_ok=True, parents=True)
            process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name)
            pil_image.save(Path(process_train_path, Path(img.name)))
        else:
            Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry").mkdir(exist_ok=True, parents=True)
            process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry")
            pil_image.save(Path(process_train_path, Path(img.name)))

    return error_list


if __name__ == '__main__':
    BASE_PATH = Path("data")
    TRAIN_DATA = Path(BASE_PATH, Path("regression", "train"))
    VALID_DATA = Path(BASE_PATH, Path("regression", "valid"))
    PROCESS_PATH = Path(BASE_PATH, Path("process"))
    retry_path = Path(PROCESS_PATH, "regression", "train", "retry")
    errors = []
    print(TRAIN_DATA)
    print(VALID_DATA)

    # for age_folder in tqdm(TRAIN_DATA.iterdir()):
    #     print(f"Cropping images of {age_folder.stem} year olds")
    #     if age_folder.is_file():
    #         errors.extend(single_dlib_face_detector(age_folder, "regression"))


    # errors.extend(dlib_face_detector([image for image in retry_path.iterdir() if image.is_file()], True))
    batch = 8
    count = 0
    images = []
    for image in retry_path.iterdir():
        images.append(image)
        count += 1
        if count >= batch:
            print("Running batch")
            errors.extend(dlib_face_detector(images, True))
            images = []
            count = 0

    print("errors")
    print(errors)
    print(len(errors))


