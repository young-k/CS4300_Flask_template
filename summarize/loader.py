"""
Create and load data.
"""

import csv
import json
import numpy as np
import os
import pandas as pd
import random
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

WORD_PATTERN = r'(?u)\b\w+\b'
END_TOKEN = 0
START_TOKEN = 1
UNKNOWN = 2


def tokenize(text):
    tokens = re.findall(WORD_PATTERN, text)
    return tokens


def create_data():
    with open('../scripts/testing.txt', 'r') as f:
        data = eval(f.read())
    comments = []
    for post in data:
        title = post['title'].replace('CMV:', '')
        op = ' '.join([title, post['body']]).replace('\n', ' ')
        for comment in post['top_comments']:
            comments.append([op, comment['body'].replace('\n', ' '), comment in post['delta_comments']])
    random.shuffle(comments)
    n_comments = len(comments)
    # cols = ['OP', 'Comment', 'Delta']
    train, test = int(0.8 * n_comments), int(0.9 * n_comments)
    with open('./data/training.txt', 'w') as f:
        f.write(str(comments[:train]))
    with open('./data/validation.txt', 'w') as f:
        f.write(str(comments[train:test]))
    with open('./data/testing.txt', 'w') as f:
        f.write(str(comments[test:]))

    # pd.DataFrame(comments[:train], columns=cols).to_csv('./data/training.csv', sep='\t', index=False)
    # pd.DataFrame(comments[train:test], columns=cols).to_csv('./data/validation.csv', sep='\t', index=False)
    # pd.DataFrame(comments[test:], columns=cols).to_csv('./data/testing.csv', sep='\t', index=False)


def create_vocab():
    data = pd.read_csv('./data/training.csv')
    documents = set()
    documents.update(data['OP'].values.astype(str).tolist())
    documents.update(data['Comment'].values.astype(str).tolist())
    vectorizer = CountVectorizer(token_pattern=WORD_PATTERN, min_df=3)
    vectorizer.fit(documents)
    vocab = {}
    for word, i in vectorizer.vocabulary_.items():
        vocab[word] = int(i+3)
    with open('./data/train_vocab.json', 'w') as f:
        json.dump(vocab, f)


def create_embedding_matrix(file_path):
    words = pd.read_table(file_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    full_matrix = words.as_matrix()
    full_vocab = {word: i for i, word in enumerate(words.index)}
    with open('./data/train_vocab.json', 'r') as f:
        sub_vocab = json.load(f)
    sub_matrix = np.zeros((len(sub_vocab) + 3, int(re.findall(r'\d+', file_path)[-1])))
    for word, i in tqdm(sub_vocab.items()):
        if word in full_vocab:
            sub_matrix[i] = full_matrix[full_vocab[word]]
    np.save('./data/train_embeddings.npy', sub_matrix)


class Loader(object):
    def __init__(self, file_path, batch_size=64, max_post=1000, max_comment=250):
        with open(file_path, 'r') as f:
            self.data = eval(f.read())
        self.mp, self.mc = max_post, max_comment
        self.batch_size = batch_size
        self.n_batches = None
        self.encodings = {'op': [], 'comments': [], 'deltas': []}
        self.batches = {'op': (), 'comments': (), 'deltas': ()}
        self.ptr = 0

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        with open('./data/train_vocab.json', 'r') as f:
            vocab = json.load(f)

        def _encode(word):
            return vocab[word] if word in vocab else UNKNOWN

        op = [sample[0] for sample in self.data]
        comments = [sample[1] for sample in self.data]
        # op, comments = self.data['OP'].values.astype(str).tolist(), self.data['Comment'].values.astype(str).tolist()
        print('Tokenizing posts...')
        for post in tqdm(op):
            tokens = tokenize(post)
            encoding = [_encode(w) for w in tokens[:self.mp] + [END_TOKEN] * (self.mp - len(tokens))]
            self.encodings['op'].append(encoding)
        print('Tokenizing comments...')
        for comment in tqdm(comments):
            tokens = tokenize(comment)
            encoding = [_encode(w) for w in tokens[:self.mc] + [END_TOKEN] * (self.mc - len(tokens))]
            self.encodings['comments'].append(encoding)
        self.encodings['deltas'] = [sample[2] for sample in self.data]

    def create_batches(self):
        self.n_batches = int(len(self.data) // self.batch_size)
        n_samples = self.n_batches * self.batch_size
        permutation = np.random.permutation(n_samples)
        self.batches['op'] = np.split(np.array(self.encodings['op'])[permutation, :], self.n_batches)
        self.batches['comments'] = np.split(np.array(self.encodings['comments'])[permutation, :], self.n_batches)
        self.batches['deltas'] = np.split(np.array(self.encodings['deltas'])[permutation], self.n_batches)

    def next_batch(self):
        self.ptr = (self.ptr + 1) % self.n_batches
        if self.ptr == 0:
            self.create_batches()
        return self.batches['op'][self.ptr], self.batches['comments'][self.ptr], self.batches['deltas'][self.ptr]
    

if __name__ == '__main__':
    glove_path = sys.argv[1]
    if not {'training.txt', 'validation.txt', 'testing.txt'}.issubset(os.listdir('./data')):
        print('Generating data...')
        create_data()
    if not os.path.exists('./data/train_vocab.json'):
        print('Generating vocab...')
        create_vocab()
    if not os.path.exists('./data/train_embeddings.npy'):
        print('Generating embedding matrix..')
        create_embedding_matrix(glove_path)
