# !/usr/bin/env python3
""" Evaluate Model"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        load = tf.train.import_meta_graph(save_path + ".meta")
        load.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        feed_dict = {x: X, y: Y}
        prediction = sess.run(y_pred, feed_dict)
        accuracy = sess.run(accuracy, feed_dict)
        loss = sess.run(loss, feed_dict)
    return prediction, accuracy, loss
