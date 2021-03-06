#!/usr/bin/env python3
""" Module to train a Neural network with mini-batch gradient descent"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Computes training with mini-batch gradient descent"""
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        feed_valid = {x: X_valid, y: Y_valid}

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            train_cost = session.run(loss, feed_dict=feed_train)
            train_accuracy = session.run(accuracy, feed_dict=feed_train)
            valid_cost = session.run(loss, feed_dict=feed_valid)
            valid_accuracy = session.run(accuracy, feed_dict=feed_valid)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            n_minibatches = X_shuffled.shape[0] // batch_size
            if X_shuffled.shape[0] % batch_size != 0:
                n_minibatches += 1

            if epoch < epochs:
                for i in range(n_minibatches):
                    X_mini = X_shuffled[i * batch_size:(i + 1) * batch_size, :]
                    Y_mini = Y_shuffled[i * batch_size:(i + 1) * batch_size, :]
                    feed_batch = {x: X_mini, y: Y_mini}
                    session.run(train_op, feed_dict=feed_batch)
                    if i % 100 == 0 and i != 0:
                        step_cost = session.run(loss, feed_dict=feed_batch)
                        step_accuracy = session.run(accuracy,
                                                    feed_dict=feed_batch)
                        print("\tStep {}:".format(i))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        save_path = saver.save(session, save_path)
    return save_path
