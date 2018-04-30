import json
import sys
import torch
import markdown2
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

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

##vaderSentiment
analyzer = SentimentIntensityAnalyzer()


def vader_agreement_score(s1,s2):
    p1 = analyzer.polarity_scores(s1.encode('utf8'))
    p2 = analyzer.polarity_scores(s2.encode('utf8'))
    return p1['compound'] - p2['compound']
    
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
    split_query = query.split('|')
    topic = split_query[0]
    statement = split_query[1]
    if not query:
        result = []
        output_message = ''
    else:
        output_message = 'Your search: ' + query
        result = topic_search(query, data, glove, dt_matrix, vocab)
        if statement != '':
            for i, r in enumerate(result):
                r['agree_score'] = abs(vader_agreement_score(statement,parsed_titles[i]))
                r['ranking_score'] = r['relevance_score'] * (1-r['agree_score'])
            result = sorted(result, key=lambda x: x['ranking_score'],reverse=True)
            print('#######################')
            print('#######################')
            print('#######################')
            print(statement)
            print('#######################')
            print('#######################')
            print('#######################')
            for r in result:
                print(r['title'],r['agree_score'],r['relevance_score'],r['ranking_score'])
        if len(result) > 0:
            titles = [res['title'] for res in result]
            parsed_titles = [r['title'].replace('CMV', '') for r in result]
            encoded_titles = model.encode(titles)
            embeds = normalize(PCA(n_components=2).fit_transform(encoded_titles))
            for i, res in enumerate(result):
                res['coordinate'] = list(embeds[i])
                res['title'] = str(res['title'])
                for comment in res['top_comments']:
                    comment['body'] = re.sub(r'$\u.*', r'&#;',str(comment['body']))
            
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
