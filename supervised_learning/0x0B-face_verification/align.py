#!/usr/bin/env python3
"""Face Align """
import dlib
import cv2
import numpy as np


class FaceAlign:
    """ Initialize Face Align """
    def __init__(self, shape_predictor_path):
        """
        shape_predictor_path is the path to the dlib shape predictor model
        Sets the public instance attributes:
            detector - contains dlib‘s default face detector
            shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        image is a numpy.ndarray of rank 3 containing an image from which
            to detect a face
        Returns: a dlib.rectangle containing the boundary box for the face
            in the image, or None on failure
        If multiple faces are detected, return the dlib.rectangle with the
            largest area
        If no faces are detected, return a dlib.rectangle that is the same
            as the image
        """
        try:
            faces = self.detector(image, 1)
            if len(faces) == 0:
                return dlib.rectangle(left=0,
                                      top=0,
                                      right=image.shape[1],
                                      bottom=image.shape[0])
            large_face = np.argmax([face.area() for face in faces])
            return faces[large_face]
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        image is a numpy.ndarray of an image from which to find facial
            landmarks
        detection is a dlib.rectangle containing the boundary box of the
            face in the image
        Returns: a numpy.ndarray of shape (p, 2)containing the landmark
            points, or None on failure
            p is the number of landmark points
            2 is the x and y coordinates of the point
        """
        try:
            pred = self.shape_predictor(image, detection)
            landmarks = [[pred.part(i).x, pred.part(i).y] for i in range(68)]
            return np.asarray(landmarks, dtype='int')
        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        image is a numpy.ndarray of rank 3 containing the image to be aligned
        landmark_indices is a numpy.ndarray of shape (3,) containing the
            indices of the three landmark points that should be used for the
            affine transformation
        anchor_points is a numpy.ndarray of shape (3, 2) containing the
            destination points for the affine transformation, scaled to the
            range [0, 1]
        size is the desired size of the aligned image
        Returns: a numpy.ndarray of shape (size, size, 3) containing the
            aligned image, or None if no face is detected
        """
        img = self.detect(image)
        landmarks = self.find_landmarks(image, img)
        points = landmarks[landmark_indices].astype('float32')
        get_affine = cv2.getAffineTransform(points, anchor_points * size)
        return cv2.warpAffine(image, get_affine, (size, size))
