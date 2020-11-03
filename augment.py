import urllib.request
import google_streetview.api
import google_streetview.helpers
import urllib
import random
import google_streetview.api
import google_streetview.helpers
from pygeodesy.ellipsoidalVincenty import LatLon
import pandas as pd
import math
from math import ceil, floor
import cv2
from PIL import Image, ImageOps
import os
import skimage.io
import skimage.transform
import skimage as sk
from skimage import util
from skimage import transform as tf


def flip_image(files, path_to_save_augmented_images):
    """
    :param files: list containing all the images along with the relative path
    :param path_to_save_augmented_images: Path where the augmented images should be stored
    :return:
    """
    j = 0
    for file in files:
        try:
            file_name = path_to_save_augmented_images + str(j) + "flip.png"
            j = j + 1
            image_obj = Image.open(file)
            rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
            rotated_image.save(file_name)
        except:
            print("Except")

def cropScaleUpScaleDown(files,path_to_save_augmented_images):
    """

        :param files: list containing all the images along with the relative path
        :param path_to_save_augmented_images: path_to_save_augmented_images: Path where the augmented images should be stored
        :return:
        """
    j=0
    for file in files:
        try:
            source = cv2.imread(file, 1)
            scaleX = 0.5
            scaleY = 0.5
            crop = source[20:500, 20:250]
            scaleDown = cv2.resize(source, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
            scaleUp = cv2.resize(source, None, fx=scaleX * 3, fy=scaleY * 3, interpolation=cv2.INTER_LINEAR)

            filename = "Images_" + str(j) + "scale.jpg"
            j += 1
            cv2.imwrite(path_to_save_augmented_images+ filename, scaleDown)
            filename = "Images_" + str(j) + "scale.jpg"
            j += 1
            cv2.imwrite(path_to_save_augmented_images + filename, scaleUp)
            filename = "Images_" + str(j) + "scale.jpg"
            j += 1
            cv2.imwrite(path_to_save_augmented_images + filename, crop)
        except:
            print("Except")

def addNoise(files, path_to_save_augmented_images):
    """

        :param files: list containing all the images along with the relative path
        :param path_to_save_augmented_images: path_to_save_augmented_images: Path where the augmented images should be stored
        :return:
        """
    j=0
    for i in files:
        try:
            images_transform = skimage.io.imread(i)
            transformed_image = sk.util.random_noise(images_transform)
            file_name = 'train_images' + str(j) + 'noise.png'
            j = j + 1
            skimage.io.imsave( path_to_save_augmented_images+ file_name, transformed_image)
        except:
            print("Except")

def randomrotate(files, path_to_save_augmented_images):
    """

        :param files: list containing all the images along with the relative path
        :param path_to_save_augmented_images: path_to_save_augmented_images: Path where the augmented images should be stored
        :return:
        """
    j=0
    random_degree = random.uniform(-50, 105)
    for i in files:
        try:
            images_transform = skimage.io.imread(i)
            rotated_image = sk.transform.rotate(images_transform, random_degree)
            file_name = 'train_images' + str(j) + 'rotate.png'
            j = j + 1
            skimage.io.imsave(path_to_save_augmented_images + file_name, rotated_image)
        except:
            print("Except")

def shearing(files, path_to_save_augmented_images):
    """

    :param files: list containing all the images along with the relative path
    :param path_to_save_augmented_images: path_to_save_augmented_images: Path where the augmented images should be stored
    :return:
    """
    j = 0
    for i in files:
        try:
            images_transform = skimage.io.imread(i)
            afine_tf = tf.AffineTransform(shear=0.5)
            modified = tf.warp(images_transform, inverse_map=afine_tf)
            file_name = 'train_images' + str(j) + 'shear.png'
            j = j + 1
            skimage.io.imsave( path_to_save_augmented_images+ file_name, modified)
        except:
            print("Except")


path_to_images_to_augment = "data/zebracrossing_newpadded/"
path_to_save_augmented_images = "data/zebracrossing_newaugmented/"
files = [path_to_images_to_augment+ f for f in os.listdir(path_to_images_to_augment)]




"""
Augmenting Images for flipping an image
"""

flip_image(files, path_to_save_augmented_images)

"""
Augmenting Images for Cropping, scaling up and scaling down of images
"""

cropScaleUpScaleDown(files, path_to_save_augmented_images)

"""
Augmenting Images for adding noise
"""
addNoise(files, path_to_save_augmented_images)

"""
Adding random rotation for Images
"""
randomrotate(files, path_to_save_augmented_images)

"""
Augmenting images by shearing
"""

shearing(files, path_to_save_augmented_images)
