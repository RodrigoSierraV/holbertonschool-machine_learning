#!/usr/bin/env python3
""" Save best model """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                save_best=False, filepath=None,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """
    learning_rate_decay is a boolean that indicates whether learning rate
        decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate
    save_best is a boolean indicating whether to save the model after each
        epoch if it is the best
    a model is considered the best if its validation loss is the lowest that
        the model has obtained
    filepath is the file path where the model should be saved
    """
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if learning_rate_decay and validation_data:
        decay = K.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(decay)
    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(early_stop)
    if save_best:
        best = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callbacks.append(best)
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          verbose=verbose,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
