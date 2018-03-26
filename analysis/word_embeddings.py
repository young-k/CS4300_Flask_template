"""
Generate word embeddings using GloVe: https://nlp.stanford.edu/projects/glove/
"""

import csv
import pandas as pd
import numpy as np
import zipfile
from sklearn.neighbors import KNeighborsClassifier


class GloVe:
    """Pre-trained GloVe vectorizer.
    Attributes:
        model: 400000 x k matrix (k = embedding dimenison)
        dict:  dictionary mapping words to index
    """
    def __init__(self, file_path):
        glove = zipfile.ZipFile(file_path, 'r')
        words = pd.read_table(glove.open('glove.6B.50d.txt'), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.model = words.as_matrix()
        self.dict = {word: i for i, word in enumerate(words.index)}
        self.clf = KNeighborsClassifier(n_neighbors=10).fit(self.model, np.zeros(self.model.shape[0]))

    def vectorize(self, word):
        if word in self.dict:
            index = self.dict[word]
            return self.model[index]
        else:
            return np.zeros(50)

    def nearest_neighbors(self, vector, n=3):
        vocab = np.array(list(self.dict.keys()))
        indices = [list(self.clf.kneighbors(vector, n_neighbors=n)[1][0])][0]
        return vocab[indices]


if __name__ == '__main__':

    model = GloVe('../data/glove.6B.zip')
    print(model.vectorize('hello'))
