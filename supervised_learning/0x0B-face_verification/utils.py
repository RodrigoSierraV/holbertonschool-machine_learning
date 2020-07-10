#!/usr/bin/env python3
""" Utilities to manage image files"""
import os
import numpy as np
import cv2
import csv


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


def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists:

    csv_path is the path to the csv to load
    params are the parameters to load the csv with
    Returns: a list of lists representing the contents found in csv_path
    """
    with open(csv_path, 'r') as csv_file:
        return [line for line in csv.reader(csv_file, params)]


def save_images(path, images, filenames):
    """
    path is the path to the directory in which the images should be saved
    images is a list/numpy.ndarray of images to save
    filenames is a list of filenames of the images to save
    Returns: True on success and False on failure
    """
    try:
        for filename, image in zip(filenames, images):
            cv2.imwrite(path+'/'+filename,
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return True
    except FileNotFoundError:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    images is a numpy.ndarray of shape (i, n, n, 3) containing the aligned
        images in the dataset
            i is the number of images
            n is the size of the aligned images
    filenames is a list of length i containing the corresponding filenames
        for images
    triplet_names is a list of length m of lists where each sublist contains
        the filenames of an anchor, positive, and negative image, respectively
            m is the number of triplets
    Returns: a list [A, P, N]
        A is a numpy.ndarray of shape (m, n, n, 3) containing the anchor
            images for all m triplets
        P is a numpy.ndarray of shape (m, n, n, 3) containing the positive
            images for all m triplets
        N is a numpy.ndarray of shape (m, n, n, 3) containing the negative
            images for all m triplets
    """
    names = [name.split('.')[0] for name in filenames]
    A = []
    P = []
    N = []
    for a, p, n in triplet_names:
        try:
            A.append(images[names.index(a)])
            P.append(images[names.index(p)])
            N.append(images[names.index(n)])
        except ValueError:
            continue
    return [np.asarray(A), np.asarray(P), np.asarray(N)]
