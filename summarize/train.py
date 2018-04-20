"""
Train TensorFlow model.
"""

import argparse
import numpy as np
import os
import tensorflow as tf
import sys
from loader import Loader
from model import RNN
from tqdm import tqdm


def train(args):
    best = np.inf
    data_tr = Loader(args.data_tr, args.batch_size, args.max_op, args.max_c)
    data_va = Loader(args.data_va, args.batch_size, args.max_op, args.max_c)

    if args.load_dir is not None:
        ckpt = tf.train.get_checkpoint_state(args.load_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    model = RNN(args)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if args.load_dir is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for ep in range(args.n_epochs):
            loss_tr = 0
            for i in tqdm(range(data_tr.n_batches)):
                posts, comments, scores = data_tr.next_batch()
                feed_dict = {model.op: posts, model.comments: comments, model.scores: scores}
                loss, _ = sess.run([model.l1_loss, model.train_step], feed_dict=feed_dict)
                loss_tr += loss
            loss_tr /= data_tr.n_batches
            print('Training accuracy: {:.3f}'.format(loss_tr))
            if (ep % 2) == 0:
                loss_va = 0
                for j in tqdm(range(data_va.n_batches)):
                    posts, comments, scores = data_va.next_batch()
                    feed_dict = {model.op: posts, model.comments: comments, model.scores: scores}
                    loss = sess.run(model.l1_loss, feed_dict=feed_dict)
                    loss_va += loss
                loss_va /= data_va.n_batches
                print('Validation accuracy: {:.3f}'.format(loss_va))
                if loss_va < best:
                    best = loss_va
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep)
                else:
                    ckpt = tf.train.get_checkpoint_state(args.save_dir)
                    saver.restore(sess, ckpt.model_checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_tr', type=str, default='./data/training.csv', help='File path to training data.')
    parser.add_argument('--data_va', type=str, default='./data/validation.csv', help='File path to validation data.')
    parser.add_argument('--load_dir', type=str, default=None, help='Directory containing pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='./models/', help='Directory in which to save trained model.')

    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for Adam optimizer.')

    parser.add_argument('--max_op', type=int, default=1000, help='Length at which to truncate posts.')
    parser.add_argument('--max_c', type=int, default=250, help='Length at which to truncate comments.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in each stage.')
    parser.add_argument('--n_units', type=int, default=128, help='Number of units in each RNN cell.')

    args = parser.parse_args(sys.argv[1:])
    train(args)
