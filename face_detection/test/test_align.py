import unittest
from face_preprocessing.align import FaceAligner
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


class MyTestCase(unittest.TestCase):
    def test_something(self):
        align = FaceAligner(True)
        im = cv2.imread('C:\\Users\\CallumDesk\\Desktop\\71\\test.png')
        plt.imshow(align(im=im))
        plt.show()

        im = cv2.imread('/home/conor/datasets/FGNET/images/001A05.JPG')
        plt.imshow(align(im))
        plt.show()

        im = cv2.imread('/home/conor/datasets/FGNET/images/001A19.JPG')
        plt.imshow(align(im))
        plt.show()


class TestBatch(unittest.TestCase):
    # def test_batch_align(self):
    #     image_folder = Path('C:\\Users\\CallumDesk\\Desktop\\71')
    #
    #     np_to_tensor = transforms.ToTensor()
    #     resized_images = [np_to_tensor(cv2.resize(cv2.imread(str(image))), dsize=(299, 299)) for image in
    #                       image_folder.iterdir()]
    #
    #
    #     # batch size, c, h, w
    #     image_batch = torch.stack(resized_images)
    #     align = FaceAligner(True)
    #     align(image_tensor_batch=image_batch)

    def test_folder_align(self):
        image_folder = Path('C:\\Uni\\DISS\\project\\data\\process\\imdb_wiki_processed')

        align = FaceAligner(True)
        out = align(image_folder=image_folder)
        plt.imshow(out[1])
        plt.show()
        plt.imshow(out[2])
        plt.show()
if __name__ == '__main__':
    unittest.main()
