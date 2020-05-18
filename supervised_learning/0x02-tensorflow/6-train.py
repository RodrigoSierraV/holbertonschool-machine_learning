#!/usr/bin/env python3
""" Module to train neural network"""
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """ Function that builds, trains, and
        saves a neural network classifier """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)
    to_save = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for iter in range(iterations + 1):
            train_acc = session.run(accuracy,
                                    feed_dict={x: X_train, y: Y_train})
            train_loss = session.run(loss, feed_dict={x: X_train, y: Y_train})
            valid_acc = session.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            valid_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            if iter % 100 == 0 or iter == iterations:
                print("After {} iterations:".format(iter))
                print("\tTraining Cost: {}".format(train_loss))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_loss))
                print("\tValidation Accuracy: {}".format(valid_acc))
            if iter < iterations:
                session.run(train_op, feed_dict={x: X_train, y: Y_train})
        return to_save.save(session, save_path)
