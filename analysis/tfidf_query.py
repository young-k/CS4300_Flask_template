import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from word_embeddings import GloVe

with open('../data/sample.json', 'r') as f:
    data = json.load(f)

documents = []
for i in range(100):
    title = data[i]['title']
    comments = ' '.join([c['text'] for c in data[i]['comments']])
    documents.append(' '.join([title, comments]))

vectorizer = TfidfVectorizer(stop_words='english', min_df=3).fit(documents)
dt_matrix = vectorizer.transform(documents).toarray()
vocab = vectorizer.get_feature_names()

for i in range(100):
    vec = dt_matrix[i]
    idx = np.argsort(-1 * vec)[:10]
    kws = [vocab[j] for j in idx]
    data[i]['keywords'] = set(kws)

model = GloVe('../data/glove.6B.zip')

print('Press Ctrl+C to quit.')
while True:
    try:
        query = input('Query: ')
        expanded = model.nearest_neighbors(query, n=10)
        relevant = [post['title'] for post in data if len(set(expanded).intersection(post['keywords'])) > 0]

        if len(relevant) == 0:
            print('No relevant posts found.')
        else:
            for title in relevant:
                print(title.replace('\n', ''))
    except KeyboardInterrupt:
        break
