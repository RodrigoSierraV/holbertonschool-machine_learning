#!/usr/bin/env python3
"""
Class to decode for machine translation
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    class that inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        vocab is an integer representing the size of the output vocabulary
        embedding is an integer representing the dimensionality of the
            embedding vector
        units is an integer representing the number of hidden units in
            the RNN cell
        batch is an integer representing the batch size
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        x is a tensor of shape (batch, 1) containing the previous word in the
            target sequence as an index of the target vocabulary
        s_prev is a tensor of shape (batch, units) containing the previous
            decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
        """
        context_vec, attention_weights = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, state
