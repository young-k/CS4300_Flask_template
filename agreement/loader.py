"""
Text data loader.
"""

import json
import pandas as pd
import numpy as np
import os
import spacy
from tqdm import tqdm

NLP = spacy.load('en')
START_TOKEN = 1
END_TOKEN = 0


def tokenize(text):
    doc = NLP(text)
    return [token.text for token in doc]


def update_vocab(file):
    if os.path.exists('./data/vocab.json'):
        with open('./data/vocab.json', 'r') as f:
            vocab = set(json.load(f).keys())
    else:
        vocab = set()
    c1 = pd.read_csv(file, index_col=0)['Comment 1'].values
    c2 = pd.read_csv(file, index_col=0)['Comment 1'].values
    for i in tqdm(c1.shape[0]):
        vocab.update(tokenize(c1[i]))
        vocab.update(tokenize(c2[i]))
    word2id = {}
    for i, word in enumerate(vocab):
        word2id[word] = i+2
    with open('./data/vocab.json', 'w') as file:
        json.dump(word2id, file)


def text2ids(text, max_length=None):
    tokens = tokenize(text)
    max_length = max_length or len(tokens)
    with open('./data/vocab.json', 'r') as file:
        word2id = json.load(file)
    return [word2id[w] for w in tokens[:max_length]] + [END_TOKEN] * (max_length - len(tokens))


class Loader(object):
    def __init__(self, filepath, batch_size=100, max_length=200):
        self.data = pd.read_csv(filepath, index_col=0)
        self.batch_size = batch_size
        self.max_length = max_length

        self.n_batches = None
        self.pointer = 0

        self.embeds1, self.embeds2, self.labels = None, None, None
        self.embeds1_b, self.embeds2_b, self.labels_b = None, None, None

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        print('Pre-processing data...')
        self.embeds1, self.embeds2, self.labels = [], [], []
        for i in tqdm(self.data.shape[0]):
            sample = self.data.iloc[i]
            self.embeds1.append(text2ids(sample['Comment 1'], max_length=self.max_length))
            self.embeds2.append(text2ids(sample['Comment 2'], max_length=self.max_length))
            self.labels.append(int(sample['Agree']))
        self.embeds1, self.embeds2, self.labels = np.array(self.embeds1), np.array(self.embeds2), np.array(self.labels)

    def create_batches(self):
        self.n_batches = int(self.data.shape[0] // self.batch_size)
        n_samples = self.n_batches * self.batch_size
        permutation = np.random.permutation(n_samples)
        self.embeds1, self.embeds2 = self.embeds1[permutation, :], self.embeds2[permutation, :]
        self.labels = self.labels[permutation]

        self.embeds1_b, self.embeds2_b = np.split(self.embeds1, self.n_batches), np.split(self.embeds2, self.n_batches)
        self.labels = np.split(self.labels, self.n_batches)

    def next_batch(self):
        self.pointer = (self.pointer + 1) % self.n_batches
        if self.pointer == 0:
            self.create_batches()
        return self.embeds1_b[self.pointer], self.embeds2_b[self.pointer], self.labels[self.pointer]


if __name__ == '__main__':
    for f in ['testing.csv', 'training.csv', 'testing.csv']:
        update_vocab(os.path.join('./data', f))
