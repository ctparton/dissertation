import cv2
import numpy as np
import face_alignment
from pathlib import Path
from PIL import Image

# Credit to Conor Turner for original implementation of this alignment method
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

class FaceAligner:
    """
    Align faces from landmarks
    """
    def __init__(self, gpu=False):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                               device='cuda' if gpu else 'cpu')

    def annotate_image(self, im):
        """
        Returns facial landmarks from face_alignment lib for a single image
        :param im: image read with OpenCV imread
        :return: array of landmarks
        """
        return self.fa.get_landmarks_from_image(im.astype(np.uint8))[0]

    def annotate_image_folder(self, image_folder):
        """
        Returns facial landmarks from face_alignment lib for a directory of images for batch processing
        :param image_folder: String dir path
        :return: dictionary mapping of images to landmarks array
        """
        return self.fa.get_landmarks_from_directory(str(image_folder))

    @staticmethod
    def rotate_image(image, angle, center):
        """
        Rotate image around center point between eyes
        :param angle rotation angle between eyes
        :param image opencv image to rotate
        :param center center point between eyes
        :return rotated image
        """
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def padded_square_crop(im, x, y, r):
        """
        Returns a crop of the rotated image
        :param im: rotated image using rotate_image
        :param x: x crop pos
        :param y: y crop pos
        :param r: crop margin
        :return: cropped image
        """
        try:
            h, w, c = im.shape
        except ValueError:
            h = im.shape[0]
            w = im.shape[1]
            im = im.reshape(h, w, 1)
            c = im.shape[2]

        if x < 0:  # left edge
            pad = np.zeros((h, -x, c), dtype=np.uint8)
            im = np.concatenate([pad, im], axis=1)
        else:
            im = im[:, x:]

        if x + r > w:  # right edge
            pad = np.zeros((im.shape[0], (x + r) - w, c), dtype=np.uint8)
            im = np.concatenate([im, pad], axis=1)
        else:
            im = im[:, :r]

        if y < 0:  # top edge
            pad = np.zeros((-y, im.shape[1], c), dtype=np.uint8)
            im = np.concatenate([pad, im], axis=0)
        else:
            im = im[y:, :]

        if y + r > h:  # bottom edge
            pad = np.zeros(((y + r) - h, im.shape[1], c), dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        else:
            im = im[:r, :]

        return im

    def align_image(self, im, kps, desired_eye_ratio=0.3, mid_eye_pos_ratio=0.45):
        """
        Align input image based on the size and center of the eyes
        :param im opencv image to rotate
        :param kps keypoints returned from annotation methods
        :param desired_eye_ratio (optional) resulting eye position ratio
        :param mid_eye_pos_ratio (optional) eye center point ratio
        """

        # kps = list(map(lambda i: np.array([kps.part(i).x, kps.part(i).y]), range(5)))
        # FAN face keypoints for eye positions
        l_eye_center = (kps[42] + kps[45]) / 2
        r_eye_center = (kps[36] + kps[39]) / 2
        # half way between eye centers
        mid_eye = (l_eye_center + r_eye_center) / 2
        eye_dist = np.sqrt(np.sum(np.square(l_eye_center - r_eye_center)))
        # angle between eyes on y axis
        angle = np.arctan2(*(l_eye_center - r_eye_center))
        # convert to degrees from x axis
        angle = 90 - np.degrees(angle)

        # rotate using mid eye center and angle computer
        rotated = self.rotate_image(im, angle, mid_eye)

        # scale eye dist to 0.5 the image width
        new_w = int(eye_dist / desired_eye_ratio)

        crop_x = int(mid_eye[0] - (new_w / 2))
        crop_y = int(mid_eye[1] - new_w * mid_eye_pos_ratio)

        return self.padded_square_crop(rotated, crop_x, crop_y, new_w)

    
    # Additions - Callum Parton
    @staticmethod
    def is_gray(img):
        """
        Checks if image is grayscale
        :param img: img to check
        :return: true if grayscale else false
        """
        # returns None if there are more than 256 colours in image
        return Image.fromarray(img).getcolors(maxcolors=256)
        # if len(img.shape) < 3: return True
        # if img.shape[2] == 1: return True
        # b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # if (b == g).all() and (b == r).all(): return True
        # return False

    @staticmethod
    # Additions - Callum Parton
    def convert_to_three_channel(img):
        """
        Converts an img to three channel representation for face_alignment annotation
        :param img: img to check
        :return: converted image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        return img2

    def __call__(self, im=None, image_folder=None, desired_eye_ratio=0.3, mid_eye_pos_ratio=0.45):
        if image_folder is not None:
            output_images = []
            # Additions - Callum Parton
            # TODO: Currently throws error if any of the images in the input folder have an alpha channel
            Path(image_folder, 'converted').mkdir(exist_ok=True)
            for img in image_folder.iterdir():
                if img.is_file() and (img.suffix == '.jpg' or img.suffix == '.png'):
                    im = cv2.imread(str(img))
                    if self.is_gray(im):
                        im = self.convert_to_three_channel(im)
                    cv2.imwrite(str(Path(image_folder, 'converted', img.name)), im)

            images_with_kps = self.annotate_image_folder(Path(image_folder, 'converted'))
            if images_with_kps is not None:
                for img, kps in images_with_kps.items():
                    if kps is not None:
                        output_images.append(self.align_image(cv2.imread(img), kps[0],
                                                              desired_eye_ratio, mid_eye_pos_ratio))
                    else:
                        output_images.append(cv2.imread(img))
            return output_images
        else:
            # Additions - Callum Parton
            # If image is already greyscale
            if len(im.shape) < 3 or im.shape[2] == 1:
                kps = self.annotate_image(im)
            else:
                kps = self.annotate_image(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

            if kps is None:
                # print('could not crop:', path)
                return im

            return self.align_image(im, kps, desired_eye_ratio, mid_eye_pos_ratio)
