"""
Train TensorFlow model.
"""

import argparse
import os
import tensorflow as tf
import sys
from loader import Loader
from model import RNN
from tqdm import tqdm


def train(args):
    best = 0
    data_tr = Loader(args.data_tr, args.batch_size, args.max_p, args.max_h)
    data_va = Loader(args.data_va, args.batch_size, args.max_p, args.max_h)

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
            acc_tr = 0
            n = data_tr.n_batches
            for i in range(data_tr.n_batches):
                premise, hypothesis, labels = data_tr.next_batch()
                feed_dict = {model.premise: premise, model.hypothesis: hypothesis, model.labels: labels}
                loss, acc, _ = sess.run([model.loss, model.accuracy, model.train_step], feed_dict=feed_dict)
                acc_tr += acc
                print('Epoch {} ({}/{}): loss={:.3f}, accuracy={:.3f}%'.format(ep+1, i+1, n, loss, 100*acc))
            acc_tr /= data_tr.n_batches
            print('Mean training accuracy: {:.3f}'.format(acc_tr))
            if (ep % 2) == 0:
                acc_va = 0
                for _ in tqdm(range(data_va.n_batches)):
                    premise, hypothesis, labels = data_va.next_batch()
                    feed_dict = {model.premise: premise, model.hypothesis: hypothesis, model.labels: labels}
                    loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
                    acc_va += acc
                acc_va /= data_va.n_batches
                print('Mean validation accuracy: {:.3f}%'.format(100*acc_va))
                if acc_va >= best:
                    best = acc_va
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep)
                else:
                    ckpt = tf.train.get_checkpoint_state(args.save_dir)
                    saver.restore(sess, ckpt.model_checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_tr', type=str, default='./data/snli_1.0_train.jsonl', help='File path to training data.')
    parser.add_argument('--data_va', type=str, default='./data/snli_1.0_dev.jsonl', help='File path to validation data.')
    parser.add_argument('--load_dir', type=str, default=None, help='Directory containing pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='./models/', help='Directory in which to save trained model.')

    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer.')

    parser.add_argument('--max_p', type=int, default=100, help='Length at which to truncate premise.')
    parser.add_argument('--max_h', type=int, default=100, help='Length at which to truncate hypothesis.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in each stage.')
    parser.add_argument('--n_units', type=int, default=128, help='Number of units in each RNN cell.')

    args = parser.parse_args(sys.argv[1:])
    train(args)
