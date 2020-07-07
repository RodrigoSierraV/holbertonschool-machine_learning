#!/usr/bin/env python3
""" Utilities to manage image files"""
import os
import numpy as np
import cv2


def load_images(images_path, as_array=True):
    """
    images_path is the path to a directory from which to load images
    as_array is a boolean indicating whether the images should be
        loaded as one numpy.ndarray
            If True, the images should be loaded as a numpy.ndarray
            of shape (m, h, w, c) where:
                m is the number of images
                h, w, and c are the height, width, and number of
                    channels of all images, respectively
            If False, the images should be loaded as a list of individual
                numpy.ndarrays
    All images are loaded in RGB format
    The images are loaded in alphabetical order by filename
    Returns: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated with each image
            in images
    """
    img_names = sorted(os.listdir(images_path))
    read_img = (cv2.imread(images_path+'/'+img) for img in img_names)
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in read_img]
    if as_array is True:
        images = np.stack(images, axis=0)
    return images, img_names
