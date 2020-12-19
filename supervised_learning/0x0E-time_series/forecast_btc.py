#!/usr/bin/env python3
"""
Script that performs forecasting
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_dataset(data, total_window_size, input_slice,
                 labels_slice, label_columns,
                 column_indices, input_width, label_width):
    """
    Function to keep together the model
    :param data: data to process
    :param total_window_size: total_window_size
    :return: dataset created for tensorflow
    """
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32)

    return ds


def compile_and_fit(model, train_ds, val_ds, patience=2):
    """
    Function to compile the model
    :param model: model to compile
    :param train_ds:
    :param val_ds:
    :param patience: patience to get of training
    :return: history of the model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model.fit(train_ds, epochs=MAX_EPOCHS,
                     validation_data=val_ds,
                     callbacks=[early_stopping])


def plot(column_indices, train_ds, model=None,
         plot_col='Close', max_subplots=3):
    """
    Function to plot many parts of the model
    :param column_indices: column_indices
    :param train_ds: train dataset
    :param model: model to plot
    :param plot_col: column to plot
    :param max_subplots: number of subplot
    :return: No return
    """
    inputs, labels = next(iter(train_ds))
    plt.figure(figsize=(12, 8))
    plot_col_index = column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel('{} [normed]'.format(plot_col))
        plt.plot(input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if label_columns:
            label_col_index = label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Close')


if __name__ == '__main__':
    preprocess = __import__('preprocess_data').preprocessor
    file = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    train_df, val_df, test_df = preprocess(file, 730)
    input_width = 24
    label_width = 24
    shift = 1
    label_columns = ['Close']
    if label_columns is not None:
        label_columns_indices = {n: i for i, n in enumerate(label_columns)}
    column_indices = {n: i for i, n in enumerate(train_df.columns)}
    total_window_size = input_width + shift
    input_slice = slice(0, input_width)
    input_indices = np.arange(total_window_size)[input_slice]
    label_start = total_window_size - label_width
    labels_slice = slice(label_start, None)
    label_indices = np.arange(total_window_size)[labels_slice]

    MAX_EPOCHS = 20
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    train_ds = make_dataset(train_df, total_window_size,
                            input_slice, labels_slice,
                            label_columns, column_indices,
                            input_width, label_width)
    val_ds = make_dataset(val_df, total_window_size,
                          input_slice, labels_slice,
                          label_columns, column_indices,
                          input_width, label_width)
    test_ds = make_dataset(test_df, total_window_size,
                           input_slice, labels_slice,
                           label_columns, column_indices,
                           input_width, label_width)
    # history = compile_and_fit(lstm_model, train_ds, val_ds)
    val_performance = {}
    performance = {}
    # val_performance['LSTM'] = lstm_model.evaluate(val_ds)
    # performance['LSTM'] = lstm_model.evaluate(test_ds, verbose=0)
    # plot(column_indices, train_ds, model=lstm_model)
