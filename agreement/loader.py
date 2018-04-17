"""
Text data loader.
"""

from __future__ import print_function
from __future__ import division

import json
import pandas as pd
import numpy as np
import os
import spacy
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from word_embeddings import GloVe

NLP = spacy.load('en')
END_TOKEN = 0
START_TOKEN = 1
UNKNOWN = 2


def tokenize(text):
    doc = NLP(text)
    return [token.text for token in doc]


def update_vocab(file):
    if os.path.exists('./data/vocab.json'):
        with open('./data/vocab.json', 'r') as f:
            vocab = set(json.load(f).keys())
    else:
        vocab = set()
    c1 = pd.read_csv(file, index_col=0)['Comment 1'].values.astype(str)
    c2 = pd.read_csv(file, index_col=0)['Comment 1'].values.astype(str)
    vec = TfidfVectorizer(min_df=2)
    vec.fit(np.concatenate([c1, c2]))
    word2id = {}
    vocab.update(list(vec.vocabulary_.keys()))
    for i, word in enumerate(vocab):
        word2id[word] = i+3
    with open('./data/vocab.json', 'w') as file:
        json.dump(word2id, file)


def embedding_matrix(word2id):
    model = GloVe('../qa_attention_ptr/ptr-net/models/glove.6B.50d.txt')
    matrix = np.zeros((len(word2id) + 3, 50))
    for word, idx in tqdm(word2id.items()):
        if word in model.dict:
            matrix[idx] = model.model[model.dict[word]]
    return matrix


def text2ids(text, max_length=None):
    tokens = tokenize(text)
    max_length = max_length or len(tokens)
    with open('./data/vocab.json', 'r') as file:
        word2id = json.load(file)
    pad = [END_TOKEN] * (max_length - len(tokens))
    ids = []
    for w in tokens[:max_length]:
        if w in word2id:
            ids.append(word2id[w])
        else:
            ids.append(UNKNOWN)
    return ids + pad


class Loader(object):
    def __init__(self, filepath, batch_size=100, max_length=200):
        self.filepath = filepath
        self.data = pd.read_csv(os.path.join(filepath, filepath.split('/')[-1] + '.csv'), index_col=0)
        self.batch_size = batch_size
        self.max_length = max_length

        self.n_batches = None
        self.pointer = 0

        self.embeds1, self.embeds2, self.labels = None, None, None
        self.embeds1_b, self.embeds2_b, self.labels_b = None, None, None

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        self.embeds1, self.embeds2, self.labels = [], [], []
        if {'embeds1.npy', 'embeds2.npy', 'labels.npy'}.issubset(os.listdir(self.filepath)):
            print('Loading pre-processed data...')
            self.embeds1 = np.load(os.path.join(self.filepath, 'embeds1.npy'))
            self.embeds2 = np.load(os.path.join(self.filepath, 'embeds2.npy'))
            self.labels = np.load(os.path.join(self.filepath, 'labels.npy'))
        else:
            print('Pre-processing data...')
            for i in tqdm(range(self.data.shape[0])):
                sample = self.data.iloc[i]
                self.embeds1.append(text2ids(sample['Comment 1'], max_length=self.max_length))
                self.embeds2.append(text2ids(sample['Comment 2'], max_length=self.max_length))
                self.labels.append(int(sample['Agree']))
            self.embeds1, self.embeds2 = np.array(self.embeds1), np.array(self.embeds2)
            self.labels = np.array(self.labels).astype(int)
            np.save(os.path.join(self.filepath, 'embeds1.npy'), self.embeds1)
            np.save(os.path.join(self.filepath, 'embeds2.npy'), self.embeds2)
            np.save(os.path.join(self.filepath, 'labels.npy'), self.labels)

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
    with open('./data/vocab.json', 'r') as file:
        word2id = json.load(file)
    matrix = embedding_matrix(word2id)
    np.save('./data/word_matrix.npy', matrix)
