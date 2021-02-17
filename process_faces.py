import face_recognition
import cv2
from pathlib import Path

def dlib_face_detector(images):
    resized_faces= [cv2.resize(cv2.imread(str(image), cv2.IMREAD_UNCHANGED), dsize=(250, 250)) for image in images]
    # plt.imshow(resized_faces[0])
    # plt.show()
    # face = cv2.imread(str(next(images)))
    # res = )
    # print(type(res))
    # cv2.imshow("reso", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # face = face_recognition.load_image_file(res)

    # plt.imshow(np.reshape(face_recognition.load_image_file(next(images))), )
    # plt.show()
    # faces =
    return face_recognition.batch_face_locations(resized_faces, batch_size=1)


if __name__ == '__main__':
    BASE_PATH = Path("data")
    TRAIN_DATA = Path(BASE_PATH, Path("train"))
    VALID_DATA = Path(BASE_PATH, Path("valid"))

    print(TRAIN_DATA)
    print(VALID_DATA)
    for age_folder in TRAIN_DATA.iterdir():
        print(f"Cropping images of {age_folder.stem} year olds")
        print(dlib_face_detector(age_folder.glob('**/*')))


