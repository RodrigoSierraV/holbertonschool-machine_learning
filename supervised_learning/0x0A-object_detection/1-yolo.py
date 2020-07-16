#!/usr/bin/env python3
""" Create class YOLO"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Class YOLO"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Instance constructor

        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
            initial filtering step
        nms_t is a float representing IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
                outputs is the number of outputs (predictions) made by the
                Darknet model
                anchor_boxes is the number of anchor boxes used for each
                    prediction
                2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:

        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        self.model = K.models.load_model(
            model_path,
            custom_objects={'GlorotUniform': K.initializers.glorot_uniform()}
        )
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the
                grid used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative
                to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        def sigmoid(x):
            """Compute sigmoid of x"""
            return 1 / (1 + np.exp(-x))
        for output in outputs:
            grid_height, grid_width, anchor_boxes, classes = output.shape
            box = np.zeros(output[:, :, :, :4].shape)
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            m_h = np.arange(grid_height).reshape(1, grid_height)
            m_h = np.repeat(m_h, grid_width, axis=0).T
            m_h = np.repeat(m_h[:, :, np.newaxis], anchor_boxes, axis=2)
            m_w = np.arange(grid_width).reshape(1, grid_width)
            m_w = np.repeat(m_w, grid_height, axis=0)
            m_w = np.repeat(m_w[:, :, np.newaxis], anchor_boxes, axis=2)

            cx = np.tile(np.arange(grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.tile(np.arange(grid_width), grid_height)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)

            pred_x = sigmoid(t_x) + cx
            pred_y = sigmoid(t_y) + cy
            pred_w = np.exp(t_w) * m_w
            pred_h = np.exp(t_h) * m_h

            norm_x = pred_x / grid_width
            norm_y = pred_y / grid_height
            norm_w = pred_w / self.model.input.shape[1].value
            norm_h = pred_h / self.model.input.shape[2].value

            top_x = (norm_x - (norm_w / 2)) * image_size[1]
            top_y = (norm_y - (norm_h / 2)) * image_size[0]
            bottom_x = (norm_x + (norm_w / 2)) * image_size[1]
            bottom_y = (norm_y + (norm_h / 2)) * image_size[0]

            box[:, :, :, 0] = top_x
            box[:, :, :, 1] = top_y
            box[:, :, :, 2] = bottom_x
            box[:, :, :, 3] = bottom_y
            boxes.append(box)

            box_conf = output[:, :, :, 4]
            conf = sigmoid(box_conf)
            conf = conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidences.append(conf)
            classes = output[:, :, :, 5:]
            probs = sigmoid(classes)
            box_class_probs.append(probs)

        return boxes, box_confidences, box_class_probs
