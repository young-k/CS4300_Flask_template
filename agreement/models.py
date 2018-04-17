"""
Agreement detection models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell

END_TOKEN = 0
START_TOKEN = 1
UNKNOWN = 2


class SiameseRNN(object):
    """A Siamese RNN for text similarity. Uses a shared bi-directional GRU followed by a fully connected net."""
    def __init__(self, cell_type=tf.contrib.rnn.GRUCell, n_layers=3, n_units=100, batch_size=100, max_length=200):
        """Constructs TensorFlow graph.

        Args:
            cell_type (method):
            n_layers (int):
            n_units (int):
            batch_size (int):
            max_length (int):
        """
        with tf.variable_scope('inputs'):
            self.input1 = tf.placeholder(tf.int32, [batch_size, max_length], 'input1')
            self.input2 = tf.placeholder(tf.int32, [batch_size, max_length], 'input2')
            self.lengths1 = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input1, END_TOKEN)), axis=1)
            self.lengths2 = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input2, END_TOKEN)), axis=1)

        with tf.variable_scope('embeddings'):
            self.word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
            self.embeds1 = tf.nn.embedding_lookup(self.word_matrix, self.input1)
            self.embeds2 = tf.nn.embedding_lookup(self.word_matrix, self.input2)

        with tf.variable_scope('gru'):
            cell_fw = _rnn_cell(cell_type, n_layers, n_units)
            cell_bw = _rnn_cell(cell_type, n_layers, n_units)
            _, st1 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embeds1, self.lengths1, dtype=tf.float32)
            _, st2 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embeds2, self.lengths2, dtype=tf.float32)
            self.out1 = tf.concat((st1[0][-1], st1[0][-1]), axis=1)
            self.out2 = tf.concat((st2[0][-1], st2[0][-1]), axis=1)

        with tf.variable_scope('loss'):
            self.labels = tf.placeholder(tf.int32, [batch_size], 'labels')
            self.prob = tf.exp(-1 * tf.norm(self.out1 - self.out2, ord=1, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.to_int32(tf.round(self.prob)), self.labels), tf.float32))
            self.loss = tf.nn.l2_loss(tf.cast(self.labels, tf.float32) - self.prob)
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)


def _rnn_cell(cell_type, n_layers, n_units, dropout_prob=0):
    """Create RNN cell."""
    def _single_cell():
        return DropoutWrapper(cell_type(n_units), output_keep_prob=1-dropout_prob)
    if n_layers == 1:
        return _single_cell()
    else:
        return MultiRNNCell([_single_cell() for _ in range(n_layers)])


if __name__ == '__main__':
    m = SiameseRNN()
