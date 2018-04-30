import json
import sys
import torch
import markdown2
import re
import numpy as np
from sklearn.decomposition import PCA

from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from analysis.word_embeddings import GloVe
from analysis.tfidf_query import find_keywords, topic_search
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from . import *


project_name = "Changing-Views"
net_id = "Yuji Akimoto (ya242), Benjamin Edwards (bje43), Jacqueline Wen (jzw22), Young Kim (yk465), Zachary Brienza (zb43)"

with open('./scripts/testing.txt', 'r') as f:
    data = eval(f.read())

data, dt_matrix, vocab = find_keywords(data, n=10)
glove = GloVe('./data/glove.6B.50d.txt')

model = torch.load('./semantics/infersent.allnli.pickle', map_location='cpu')
model.set_glove_path('./data/glove.840B.300d.txt')
model.build_vocab_k_words(K=100000)

# vaderSentiment
analyzer = SentimentIntensityAnalyzer()

@irsystem.route('/', methods=['GET'])
def home():
  query = request.args.get('search')
  if not query:
    result = []
    output_message = ''
  else:
    output_message = 'Your search: ' + query
    result = topic_search(query, data, glove, dt_matrix, vocab)
  return render_template('home.html', name=project_name, net_id=net_id, output_message=output_message, data=result)

def unicode_replace(string):
    string = string.replace('/u', '&#')
    idxs = [i for i, j in enumerate(string) if j == '#']
    orig_len = len(string)
    ctr = 0
    for elt in idxs:
        if orig_len == elt + 5:
            string += ';'
        else:
            string = string[0:(elt + 5 + ctr)] + ';' + string[(elt + 5 + ctr):]
            ctr += 1
    return(string)

@irsystem.route('results', methods=['GET'])
def search():
    query = request.args.get('search')
    topic = query
    statement = request.args.get('opinion', '')
    if not query:
        result = []
        output_message = ''
    else:
        output_message = 'Your search: ' + query
        result = topic_search(topic, data, glove, dt_matrix, vocab)
        if len(result) > 0:
            # VADER RANKING
            if statement != '':
                parsed_titles = [r['title'] for r in result]
                statement_sentiment = analyzer.polarity_scores(statement.encode('utf8'))['compound']
                for i, r in enumerate(result):
                    r['agree_score'] = abs(statement_sentiment-analyzer.polarity_scores(parsed_titles[i].encode('utf8'))['compound'])
                    r['ranking_score'] = r['relevance_score'] * (1 - r['agree_score'])
                result = sorted(result, key=lambda x: x['ranking_score'],reverse=True)
            titles = [res['title'] for res in result] + [statement]
            encoded_titles = model.encode(titles)
            embeds = np.reshape(PCA(n_components=2).fit_transform(encoded_titles), (-1, 2))
            for i, res in enumerate(result):
                res['coordinate'] = [float(embeds[i, 0]), float(embeds[i, 1])]
                res['title'] = str(res['title'])
                for comment in res['top_comments']:
                    comment['body'] = unicode_replace(comment['body'])

            for post in data:
                words = post['keywords']
                post['keywords'] = list(words)
                author = post['author']
                post['top_comments'] = list(filter(lambda x: x['author']!=author, post['top_comments']))
                post['body'] = markdown2.markdown(post['body'])
                for comment in post['top_comments']:
                    if comment in post['delta_comments']:
                        comment['ranking_score'] = 5 * comment['score']
                    else:
                        comment['ranking_score'] = comment['score']
                post['top_comments'] = sorted(post['top_comments'], key=lambda x: x['ranking_score'],reverse=True)
                for comment in post['top_comments'][:5]:
                    comment['html_body'] = markdown2.markdown(comment['body'])

    return render_template('search.html', name=project_name, query=query, output_message=output_message, data=result, 
                           opinion_coor=[float(embeds[-1, 0]), float(embeds[-1, 1])])
