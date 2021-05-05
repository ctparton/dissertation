import face_recognition
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm import tqdm
from face_preprocessing.align import FaceAligner
import os


def prepare_dir_for_alignment(image_dir):
    """
    Splits a directory into smaller chunks to ease processing
    :param image_dir: root directory of images to split
    :return: path to the created dir
    """
    Path(PROCESS_PATH, Path(f"image_split_{image_dir.name}")).mkdir(exist_ok=True, parents=True)
    folder_size = 2005
    images = []
    count = 0
    for image in image_dir.iterdir():
        if image.is_file() and image.suffix != '.csv':
            images.append(image)
        if len(images) >= folder_size:
            for im in images:
                Path(PROCESS_PATH, Path(f"image_split_{image_dir.name}"), Path(f"{count}")).mkdir(exist_ok=True,
                                                                                                  parents=True)
                shutil.copy(im, Path(PROCESS_PATH, Path(f"image_split_{image_dir.name}"), Path(f"{count}"),
                                     Path(f"{im.name}")))
            images = []
            print(f"Folder {count} copied successfully")
            count += 1
        else:
            pass

    return Path(PROCESS_PATH, Path(f"image_split_{image_dir.name}"))


def align_dir(image_dir):
    """
    Aligns images in a directory

    :param image_dir: directory of images to align
    :return: void
    """
    retry_count = 3
    has_subfolders = False
    align = FaceAligner(True)
    output_dir_name = f'aligned_{image_dir.name}'
    Path(PROCESS_PATH, output_dir_name).mkdir(exist_ok=True, parents=True)
    for folder in image_dir.iterdir():
        if folder.is_dir() and folder.name != 'converted':
            has_subfolders = True
            print(f"aligning {folder}")
            image_folder = folder
            out = align(image_folder=image_folder)
            print(f"Writing images")
            for im in tqdm(range(len(out))):
                try:
                    cv2.imwrite(str(Path(PROCESS_PATH, output_dir_name, list(folder.iterdir())[im].name)), out[im])
                except:
                    print(f"CV2 error occurred")

    if not has_subfolders:
        print("Aligning on root files")
        out = align(image_folder=image_dir)
        print(f"Writing images")
        for im in tqdm(range(len(out))):
            try:
                cv2.imwrite(str(Path(PROCESS_PATH, output_dir_name, list(folder.iterdir())[im].name)), out[im])
            except Exception as E:
                print(f"CV2 error occurred {E}")


def align_images(image):
    """
    Aligns a single image and writes it to a dir
    :param image: the image to align
    :return: void
    """

    align = FaceAligner(gpu=True)
    im = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    new_im = align(im)
    # cv2.imshow('image', im)
    # cv2.imshow('new_image', new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(str(Path(PROCESS_PATH, 'aligned_imdb_wiki_processed', image.name)), new_im)


def dlib_face_detector(images, retry):
    """
    Applies the dlib CNN based face detector on a batch of images
    :param images: a batch of images
    :param retry: amount of times to retry if detection fails
    :return: A list of the images where detection failed
    """
    error_list = []
    image_back = images
    print(images)
    print("Resizing faces")
    resized_faces = [cv2.resize(cv2.imread(str(image), cv2.IMREAD_UNCHANGED)[..., ::-1], dsize=(299, 299)) for image in
                     images]
    print("Getting face locations")
    # could increase batch size to speed up processing
    face_locations = face_recognition.batch_face_locations(resized_faces, batch_size=64)
    for i in range(len(resized_faces)):
        print(f"resized_faces {len(resized_faces)}")
        print(f"locations {len(face_locations)}")
        print(f"images {len(image_back)}")
        if not face_locations[i]:
            error_list.append(images[i])
            pil_image = Image.fromarray(resized_faces[i])
            Path(PROCESS_PATH, Path(images[i].parent.name), "retry").mkdir(exist_ok=True, parents=True)
            pil_image.save(Path(PROCESS_PATH, Path(images[i].parent.name), Path("retry"), Path(images[i].name)))
        else:
            # bounding box regions
            print(face_locations[i][0])
            # get the bounding box
            top, right, bottom, left = face_locations[i][0]
            image = resized_faces[i]
            Path(PROCESS_PATH, Path(images[i].parent.name), "loose_crop").mkdir(exist_ok=True, parents=True)
            Path(PROCESS_PATH, Path(images[i].parent.name), "very_loose_crop_crop").mkdir(exist_ok=True, parents=True)
            try:
                try:
                    # produce the crops
                    loose_crop = image[top - int((top / 1.5)):bottom + int((top / 1.5)),
                                 left - int((top / 1.5)):right + int((top / 1.5))]
                    loose_crop = Image.fromarray(loose_crop)
                    loose_crop.save(
                        Path(PROCESS_PATH, Path(images[i].parent.name), Path("very_loose_crop_crop"),
                             Path(images[i].name)))
                except (ValueError, SystemError):
                    try:
                        loose_crop = image[top - int((top / 2)):bottom + int((top / 2)),
                                     left - int((top / 2)):right + int((top / 2))]
                        loose_crop = Image.fromarray(loose_crop)
                        loose_crop.save(
                            Path(PROCESS_PATH, Path(images[i].parent.name), Path("very_loose_crop_crop"),
                                 Path(images[i].name)))
                    except (ValueError, SystemError):
                        loose_crop = image[top - int((top / 3)):bottom + int((top / 3)),
                                     left - int((top / 3)):right + int((top / 3))]
                        loose_crop = Image.fromarray(loose_crop)
                        loose_crop.save(
                            Path(PROCESS_PATH, Path(images[i].parent.name), Path("very_loose_crop_crop"),
                                 Path(images[i].name)))
                tighter_crop = image[top - int((top / 6)):bottom + int((top / 6)),
                               left - int((top / 6)):right + int((top / 6))]
                tighter_crop = Image.fromarray(tighter_crop)

                tighter_crop.save(
                    Path(PROCESS_PATH, Path(images[i].parent.name), Path("loose_crop"), Path(images[i].name)))
            except (ValueError, SystemError):
                error_list.append(images[i])
                pil_image = Image.fromarray(resized_faces[i])
                Path(PROCESS_PATH, Path(images[i].parent.name), "retry").mkdir(exist_ok=True, parents=True)
                pil_image.save(Path(PROCESS_PATH, Path(images[i].parent.name), Path("retry"), Path(images[i].name)))
    return error_list


def single_dlib_face_detector(img, mode):
    """
    Applies the CNN based dlib face detector on a single image

    :param img: img to crop
    :param mode: determines saved location
    :return: error list if the face detection fails
    """
    print(img.name)
    error_list = []
    image = face_recognition.load_image_file(img)
    try:
        face_locations = face_recognition.face_locations(image, model="cnn")
        print(face_locations)
        if not face_locations:
            error_list.append(img)
            pil_image = Image.fromarray(image)
            Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry").mkdir(exist_ok=True, parents=True)
            process_train_path = Path(PROCESS_PATH, img.parent.parent.name, img.parent.name, "retry")
            pil_image.save(Path(process_train_path, Path(img.name)))
        else:
            top, right, bottom, left = face_locations[0]
            # plt.imshow(image)
            # plt.show()
            test_original_image = image[top:bottom, left:right]
            Image.fromarray(test_original_image).save(Path(PROCESS_PATH, Path("test.jpg")))
            # plt.imshow(test_original_image)
            # plt.show()

            # face_image = image[int(top/2):int((bottom+image.shape[0]) / 2), int(left/2):int((right+image.shape[1]) / 2)]
            loose_crop = image[top - int((top / 2)):bottom + int((top / 2)),
                         left - int((top / 2)):right + int((top / 2))]

            # plt.imshow(loose_crop)
            # plt.show()
            # tighter_crop = image[top - int((top / 6)):bottom + int((top / 6)),
            #              left - int((top / 6)):right + int((top / 6))]
            # plt.imshow(tighter_crop)
            # plt.show()

            try:
                try:
                    loose_crop = image[top - int((top / 1.5)):bottom + int((top / 1.5)),
                                 left - int((top / 1.5)):right + int((top / 1.5))]
                    loose_crop = Image.fromarray(loose_crop)
                    loose_crop.save(
                        Path(PROCESS_PATH, Path(img.parent.name), Path("very_loose_crop_crop"),
                             Path(img.name)))
                except (ValueError, SystemError):
                    try:
                        loose_crop = image[top - int((top / 2)):bottom + int((top / 2)),
                                     left - int((top / 2)):right + int((top / 2))]
                        loose_crop = Image.fromarray(loose_crop)
                        loose_crop.save(
                            Path(PROCESS_PATH, Path(img.parent.name), Path("very_loose_crop_crop"),
                                 Path(img.name)))
                    except (ValueError, SystemError):
                        loose_crop = image[top - int((top / 3)):bottom + int((top / 3)),
                                     left - int((top / 3)):right + int((top / 3))]
                        loose_crop = Image.fromarray(loose_crop)
                        loose_crop.save(
                            Path(PROCESS_PATH, Path(img.parent.name), Path("very_loose_crop_crop"),
                                 Path(img.name)))
                tighter_crop = image[top - int((top / 6)):bottom + int((top / 6)),
                               left - int((top / 6)):right + int((top / 6))]
                tighter_crop = Image.fromarray(tighter_crop)

                Path(PROCESS_PATH, Path(img.parent.name), "loose_crop").mkdir(exist_ok=True, parents=True)
                Path(PROCESS_PATH, Path(img.parent.name), "very_loose_crop_crop").mkdir(exist_ok=True, parents=True)

                tighter_crop.save(
                    Path(PROCESS_PATH, Path(img.parent.name), Path("loose_crop"), Path(img.name)))
            except (ValueError, SystemError):
                error_list.append(img)
                pil_image = Image.fromarray(img)
                Path(PROCESS_PATH, Path(img.parent.name), "retry").mkdir(exist_ok=True, parents=True)
                pil_image.save(Path(PROCESS_PATH, Path(img.parent.name), Path("retry"), Path(img.name)))

    except Exception as e:
        print(e)
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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    BASE_PATH = Path("../data")
    TRAIN_DATA = Path(BASE_PATH, Path("process"), Path("regression", "train"))
    VALID_DATA = Path(BASE_PATH, Path("process"), Path("regression", "valid"))
    TESTING_DATA = Path(BASE_PATH, Path("test"))
    IMDB_WIKI_DATA = Path(BASE_PATH, Path("process"), Path("image_split_imdb_wiki_processed"))
    PROCESS_PATH = Path(BASE_PATH, Path("process"))
    # PROCESS_PATH = Path(BASE_PATH, Path("process_new"), Path("regression"))
    retry_path = Path(PROCESS_PATH, "regression", "train", "retry")
    errors = []
    print(TRAIN_DATA)
    print(VALID_DATA)
    print(IMDB_WIKI_DATA)

    for age_folder in tqdm(TESTING_DATA.iterdir()):
        print(f"Cropping images of {age_folder.stem} year olds")
        if age_folder.is_file():
            errors.extend(single_dlib_face_detector(age_folder, "regression"))

    batch_size = 64
    images = []
    for image in TRAIN_DATA.iterdir():
        if image.is_file() and image.suffix != '.csv':
            images.append(image)
        if len(images) > batch_size:
            print("Running batch")
            errors.extend(dlib_face_detector(images, False))
            print("batch complete")
            images = []
        else:
            print(len(images))
    count = 0

    images = []
    for image in VALID_DATA.iterdir():
        if image.is_file() and image.suffix != '.csv':
            images.append(image)
        if len(images) > batch_size:
            print("Running batch")
            errors.extend(dlib_face_detector(images, False))
            print("batch complete")
            images = []
        else:
            print(len(images))
    count = 0

    for age_folder in tqdm(VALID_DATA.iterdir()):
        print(f"Cropping images of {age_folder.stem} year olds")
        if age_folder.is_file():
            errors.extend(single_dlib_face_detector(age_folder, "regression"))

    for image in Path(PROCESS_PATH, 'imdb_wiki_processed').iterdir():
        print(f"Aligning image {count}")
        print(f"File name {image.name}")
        align_images(image)
        count += 1

    print("aligning image dir")
    prepare_dir_for_alignment(Path(IMDB_WIKI_DATA))
    align_dir(Path(IMDB_WIKI_DATA))

    print("Errors")
    print(errors)
    print(len(errors))
