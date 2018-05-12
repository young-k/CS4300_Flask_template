from __future__ import print_function
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from word_embeddings import GloVe


def find_keywords(data, n=10):
    documents = []
    for i, sample in enumerate(data):
        title = sample['title']
        comments = ' '.join(c['body'] for c in sample['top_comments'])
        documents.append(' '.join([title, comments]))

    vectorizer = TfidfVectorizer(min_df=3, stop_words='english').fit(documents)
    dt_matrix = vectorizer.transform(documents).toarray()
    vocab = vectorizer.get_feature_names()
    argsort = np.argsort(-1 * dt_matrix, axis=1)

    for j, sample in enumerate(data):
        keywords = [vocab[j] for j in argsort[j, :n]]
        sample['keywords'] = keywords
    return data, dt_matrix, vectorizer.vocabulary_


def topic_search(keyword, data, model, dt_matrix, vocab):
    expanded = [keyword] + model.nearest_neighbors(keyword, n=3)
    original_weighting = 5

    def _is_relevant(sample_post):
        return len(set(expanded).intersection(sample_post['keywords'])) > 0

    def _relevance_score(sample_post):
        overlap = set(expanded).intersection(sample_post['keywords'])
        def is_original(word):
            return  1 + original_weighting * int(word == keyword)
        score = sum([is_original(word) * dt_matrix[data.index(sample_post), vocab[word]] for word in overlap])
        return score
    
    relevant = [post for post in data if _is_relevant(post)]
    for post in relevant:
        post['relevance_score'] = _relevance_score(post)
    relevant = sorted(relevant, key=_relevance_score, reverse=True)
    return relevant


if __name__ == '__main__':
    with open('../data/data.json', 'r') as f:
        sample_data = json.load(f)

    sample_data, doc_term_matrix, vocab_dict = find_keywords(sample_data)
    glove = GloVe('../data/glove.6B.50d.txt')

    print('Press Ctrl+C to quit.')
    while True:
        try:
            query = raw_input('Query: ')
            relevant_posts = topic_search(query, sample_data, glove, doc_term_matrix, vocab_dict)

            if len(relevant_posts) == 0:
                print('No relevant posts found.')
            else:
                for post in relevant_posts:
                    print(post['title'].replace('\n', ''))
        except KeyboardInterrupt:
            break
