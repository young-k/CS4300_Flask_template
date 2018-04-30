import json
import sys
import torch
import markdown2
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from analysis.word_embeddings import GloVe
from analysis.tfidf_query import find_keywords, topic_search
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

@irsystem.route('results', methods=['GET'])
def search():
    query = request.args.get('search')
    if not query:
        result = []
        output_message = ''
    else:
        output_message = 'Your search: ' + query
        result = topic_search(query, data, glove, dt_matrix, vocab)
        if len(result) > 0:
            titles = [res['title'] for res in result]
            encoded_titles = model.encode(titles)
            embeds = normalize(PCA(n_components=2).fit_transform(encoded_titles))
            for i, res in enumerate(result):
                res['coordinate'] = list(embeds[i])
                res['title'] = str(res['title'])
                for comment in res['top_comments']:
                    comment['body'] = str(comment['body'])
            
            for post in data:
                words = post['keywords']
                post['keywords'] = list(words)
                for comment in post['top_comments']:
                    
                    if comment in post['delta_comments']:
                        comment['ranking_score'] = 5 * comment['score']
                    else:
                        comment['ranking_score'] = comment['score']
                post['top_comments'] = sorted(post['top_comments'], key=lambda x: x['ranking_score'],reverse=True)
                for comment in post['top_comments'][:5]:
                    comment['html_body']= markdown2.markdown(comment['body'])
    return render_template('search.html', name=project_name, query=query, output_message=output_message, data=result)
