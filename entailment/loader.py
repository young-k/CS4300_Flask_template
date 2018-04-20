"""
Create and load data.
"""

import csv
import json
import numpy as np
import os
import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

WORD_PATTERN = r'(?u)\b\w+\b'
END_TOKEN = 0
UNKNOWN = 1


def tokenize(text):
    tokens = re.findall(WORD_PATTERN, text)
    return tokens


def create_vocab():
    with open('./data/snli_1.0_train.jsonl', 'r') as f:
        data = [eval(l) for l in f.readlines()]
    premises = [sample['sentence1'] for sample in data]
    hypotheses = [sample['sentence2'] for sample in data]
    vectorizer = CountVectorizer(token_pattern=WORD_PATTERN, min_df=3)
    vectorizer.fit(premises + hypotheses)
    vocab = {}
    for word, i in vectorizer.vocabulary_.items():
        vocab[word] = int(i+2)
    with open('./data/train_vocab.json', 'w') as f:
        json.dump(vocab, f)


def create_embedding_matrix(file_path):
    words = pd.read_table(file_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    full_matrix = words.as_matrix()
    full_vocab = {word: i for i, word in enumerate(words.index)}
    with open('./data/train_vocab.json', 'r') as f:
        sub_vocab = json.load(f)
    sub_matrix = np.zeros((len(sub_vocab) + 2, int(re.findall(r'\d+', file_path)[-1])))
    for word, i in tqdm(sub_vocab.items()):
        if word in full_vocab:
            sub_matrix[i] = full_matrix[full_vocab[word]]
    np.save('./data/train_embeddings.npy', sub_matrix)


class Loader(object):
    def __init__(self, file_path, batch_size=64, max_premise=82, max_hypothesis=62):
        with open(file_path, 'r') as f:
            self.data = [eval(l) for l in f.readlines()]
        self.max_p, self.max_h = max_premise, max_hypothesis
        self.batch_size = batch_size
        self.n_batches = None
        self.encodings = None
        self.batches = {'premise': (), 'hypothesis': (), 'label': ()}
        self.ptr = 0

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        self.encodings = {'premise': [], 'hypothesis': [], 'label': []}
        with open('./data/train_vocab.json', 'r') as f:
            vocab = json.load(f)

        def _encode(word):
            return vocab[word] if word in vocab else UNKNOWN

        label2int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        premises = [sample['sentence1'] for sample in self.data if sample['gold_label'] != '-']
        hypotheses = [sample['sentence2'] for sample in self.data if sample['gold_label'] != '-']
        labels = [label2int[sample['gold_label']] for sample in self.data if sample['gold_label'] != '-']
        print('Tokenizing premises...')
        for sentence in tqdm(premises):
            tokens = tokenize(sentence)
            encoding = [_encode(w) for w in tokens[:self.max_p] + [END_TOKEN] * (self.max_p - len(tokens))]
            self.encodings['premise'].append(encoding)
        print('Tokenizing hypotheses...')
        for sentence in tqdm(hypotheses):
            tokens = tokenize(sentence)
            encoding = [_encode(w) for w in tokens[:self.max_h] + [END_TOKEN] * (self.max_h - len(tokens))]
            self.encodings['hypothesis'].append(encoding)
        self.encodings['premise'] = np.array(self.encodings['premise'])
        self.encodings['hypothesis'] = np.array(self.encodings['hypothesis'])
        self.encodings['label'] = np.array(labels)

    def create_batches(self):
        self.n_batches = int(len(self.data) // self.batch_size)
        n_samples = self.n_batches * self.batch_size
        permutation = np.random.permutation(n_samples)
        self.batches['premise'] = np.split(self.encodings['premise'][permutation, :], self.n_batches)
        self.batches['hypothesis'] = np.split(self.encodings['hypothesis'][permutation, :], self.n_batches)
        self.batches['label'] = np.split(self.encodings['label'][permutation], self.n_batches)

    def next_batch(self):
        self.ptr = (self.ptr + 1) % self.n_batches
        if self.ptr == 0:
            self.create_batches()
        return self.batches['premise'][self.ptr], self.batches['hypothesis'][self.ptr], self.batches['label'][self.ptr]
    

if __name__ == '__main__':
    glove_path = sys.argv[1]
    if not os.path.exists('./data/train_vocab.json'):
        print('Generating vocab...')
        create_vocab()
    if not os.path.exists('./data/train_embeddings.npy'):
        print('Generating embedding matrix..')
        create_embedding_matrix(glove_path)
