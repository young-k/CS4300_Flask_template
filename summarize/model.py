"""
Attention-based RNN for score prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.contrib.seq2seq import AttentionWrapper, BasicDecoder, LuongAttention, TrainingHelper
from tensorflow.contrib.seq2seq import dynamic_decode

END_TOKEN = 0
START_TOKEN = 1
UNKNOWN = 2


def _rnn_cell(n_layers, n_units, cell_fn=GRUCell):
    if n_layers == 1:
        return cell_fn(n_units)
    else:
        return MultiRNNCell([cell_fn(n_units) for _ in range(n_layers)])


class RNN(object):
    def __init__(self, args):
        with tf.variable_scope('inputs'):
            self.op = tf.placeholder(tf.int32, [args.batch_size, args.max_op])
            self.comments = tf.placeholder(tf.int32, [args.batch_size, args.max_c])
            lengths_p = tf.reduce_sum(tf.to_int32(tf.not_equal(self.op, END_TOKEN)), axis=1)
            lengths_c = tf.reduce_sum(tf.to_int32(tf.not_equal(self.comments, END_TOKEN)), axis=1)

        with tf.variable_scope('embeddings'):
            word_matrix = tf.Variable(np.load('./data/train_embeddings.npy'), dtype=tf.float32, trainable=True)
            embeds_p = tf.nn.embedding_lookup(word_matrix, self.op)
            embeds_c = tf.nn.embedding_lookup(word_matrix, self.comments)

        with tf.variable_scope('p_context'):
            cell_fw, cell_bw = _rnn_cell(args.n_layers, args.n_units), _rnn_cell(args.n_layers, args.n_units)
            p_context, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embeds_p, sequence_length=lengths_p,
                                                           dtype=tf.float32)
            p_context = tf.concat(p_context, axis=2)

        with tf.variable_scope('c_context'):
            cell_fw, cell_bw = _rnn_cell(args.n_layers, args.n_units), _rnn_cell(args.n_layers, args.n_units)
            c_context, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embeds_c, sequence_length=lengths_c,
                                                           dtype=tf.float32)
            c_context = tf.concat(c_context, axis=2)

        with tf.variable_scope('p2c_alignment'):
            attn = LuongAttention(args.n_units, p_context, memory_sequence_length=lengths_p)
            attn_cell = AttentionWrapper(_rnn_cell(args.n_layers, args.n_units), attn, alignment_history=True)
            attn_state = attn_cell.zero_state(args.batch_size, dtype=tf.float32)
            helper = TrainingHelper(c_context, lengths_c)
            decoder = BasicDecoder(attn_cell, helper, attn_state)
            aligned, state, _ = dynamic_decode(decoder, maximum_iterations=args.max_c)
            self.attn_weights = tf.transpose(state.alignment_history.stack(), [1, 0, 2])

        with tf.variable_scope('scores'):
            cell = _rnn_cell(args.n_layers, args.n_units)
            _, final_state = tf.nn.dynamic_rnn(cell, aligned.rnn_output, lengths_c, dtype=tf.float32)
            logits = tf.layers.dense(tf.concat(final_state, axis=1), 2)
            self.preds = tf.nn.softmax(logits, axis=1)

        with tf.variable_scope('loss'):
            self.labels = tf.placeholder(tf.int32, [args.batch_size])
            targets = tf.one_hot(self.labels, depth=2)
            self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets, logits, 100))
            self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            correct = tf.equal(tf.argmax(self.preds, axis=1, output_type=tf.int32), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
