"""
Generate word embeddings using GloVe: https://nlp.stanford.edu/projects/glove/
"""

import csv
import numpy as np
import os
import pandas as pd
import subprocess
from sklearn.neighbors import KNeighborsClassifier


class GloVe:
    """Pre-trained GloVe vectorizer.
    Attributes:
        model: 400000 x k matrix (k = embedding dimension)
        dict:  dictionary mapping words to index
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print('Downloading pre-trained word vectors')
            subprocess.check_call(['bash', '-c', 'wget -P ../data/ http://nlp.stanford.edu/data/glove.6B.zip'])
        print('Loading word embeddings...')
        words = pd.read_table(file_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.model = words.as_matrix()
        self.dict = {word: i for i, word in enumerate(words.index)}
        self.clf = KNeighborsClassifier(n_neighbors=10).fit(self.model, np.zeros(self.model.shape[0]))
        self.vocab = np.array(sorted(self.dict, key=lambda k: self.dict[k]))

    def vectorize(self, word):
        if word in self.dict:
            index = self.dict[word]
            return self.model[index]
        else:
            return np.zeros(50)

    def nearest_neighbors(self, query, n=3):
        if type(query) == str:
            vector = self.vectorize(query).reshape(1, -1)
        elif type(query) == np.ndarray:
            vector = query.reshape(1, -1)
        else:
            return None
        indices = [list(self.clf.kneighbors(vector, n_neighbors=n)[1][0])][0]
        return list(self.vocab[indices])


if __name__ == '__main__':

    model = GloVe('../data/glove.6B.zip')
    print(model.vectorize('hello'))
