import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from word_embeddings import GloVe

with open('../data/sample.json', 'r') as f:
    sample_data = json.load(f)


def find_keywords(data, n=10):
    documents = []
    for i, sample in enumerate(data):
        title = sample['title']
        comments = ' '.join(c['text'] for c in sample['comments'])
        documents.append(' '.join([title, comments]))

    vectorizer = TfidfVectorizer(min_df=3, stop_words='english').fit(documents)
    dt_matrix = vectorizer.transform(documents).toarray()
    vocab = vectorizer.get_feature_names()
    argsort = np.argsort(-1 * dt_matrix, axis=1)

    for j, sample in enumerate(data):
        keywords = [vocab[j] for j in argsort[j, :n]]
        sample['keywords'] = set(keywords)
    return data


sample_data = find_keywords(sample_data)
glove = GloVe('../data/glove.6B.zip')


def topic_search(keyword, data, model):
    expanded = model.nearest_neighbors(keyword, n=10)
    relevant = [post for post in data if len(set(expanded).intersection(post['keywords'])) > 0]
    return relevant


if __name__ == '__main__':
    print('Press Ctrl+C to quit.')
    while True:
        try:
            query = input('Query: ')
            relevant_posts = topic_search(query, sample_data, glove)

            if len(relevant_posts) == 0:
                print('No relevant posts found.')
            else:
                for post in relevant_posts:
                    print(post['title'].replace('\n', ''))
        except KeyboardInterrupt:
            break
