import json
import sys
import os
import spacy
import torch
import numpy as np

from torch.autograd import Variable
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from analysis.word_embeddings import GloVe
from analysis.tfidf_query import find_keywords, topic_search
from entailment.text_field_embedder import TextFieldEmbedder
from entailment.feedforward import FeedForward
from entailment.similarity_function import SimilarityFunction
from entailment.elmo_token_embedder import ElmoTokenEmbedder
from entailment.model import DecomposableAttention
from entailment.elmo_indexer import ELMoTokenCharactersIndexer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from . import *


project_name = "Changing-Views"
net_id = "Yuji Akimoto (ya242), Benjamin Edwards (bje43), Jacqueline Wen (jzw22), Young Kim (yk465), Zachary Brienza (zb43)"

with open('./data/data.json', 'r') as f:
    data = json.load(f)

data, dt_matrix, vocab = find_keywords(data, n=10)
glove = GloVe('./data/glove.6B.50d.txt')

#sentiment stuff 
with open('entailment/models/config.json', 'r') as f:
    config = json.load(f)
    
with open('entailment/models/vocabulary/labels.txt', 'r') as f:
    sentiment_vocab = {word: i for i, word in enumerate(f.readlines())}
    
serialization_dir = 'entailment/models'
weights_file = 'entailment/models/weights.th'
params = config.pop('model')
model_type = params.pop('type')
initializers = params.pop('initializer')
embedder_params = params.pop('text_field_embedder')
token_embedders = {}
for key, params2 in embedder_params.items():
    token_embedders[key] = ElmoTokenEmbedder(**params2)
text_field_embedder = TextFieldEmbedder(token_embedders)
attend_ff_params = params.pop('attend_feedforward')
attend_feedforward = FeedForward(**attend_ff_params)
similarity_func_params = params.pop('similarity_function')
similarity_function = SimilarityFunction()
compare_ff_params = params.pop('compare_feedforward')
compare_feedforward = FeedForward(**compare_ff_params)
aggregate_ff_params = params.pop('aggregate_feedforward')
aggregate_feedforward = FeedForward(**aggregate_ff_params)
model = DecomposableAttention(sentiment_vocab, text_field_embedder, attend_feedforward,
                              similarity_function, compare_feedforward,
                              aggregate_feedforward)
model.load(serialization_dir, weights_file)
nlp = spacy.load('en')
indexer = ELMoTokenCharactersIndexer()

##vaderSentiment
analyzer = SentimentIntensityAnalyzer()

def tokenize(text):
    doc = nlp(text)
    return [token for token in doc]
    
def agreement_score(premises, hypothesis):
    n_premises = len(premises)
    p_array = []
    for premise in premises:
        p_tokens = tokenize(premise)
        p_array.append([indexer.token_to_indices(t, None) for t in p_tokens])
    h_tokens = tokenize(hypothesis)
    h_array = [[indexer.token_to_indices(t, None) for t in h_tokens] * n_premises]
    premise = {'elmo': Variable(torch.Tensor(p_array).type(torch.LongTensor))}
    hypothesis = {'elmo': Variable(torch.Tensor(h_array).type(torch.LongTensor))}
    outputs = model.forward(premise, hypothesis)
    preds = 100 * np.squeeze(outputs['label_probs'].data.numpy())
    attn = np.squeeze(outputs['p2h_attention'].data.numpy())
    return preds, attn
    
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
        output_message = 'Your search: ' + topic
        result = topic_search(topic, data, glove, dt_matrix, vocab)
        parsed_titles = [r['title'].replace('CMV', '') for r in result]
        
        if statement != '':
            #agree_scores, attns = agreement_score(parsed_titles, statement)
            for i, r in enumerate(result):
                r['agree_score'] = abs(vader_agreement_score(statement,parsed_titles[i]))
                r['ranking_score'] = r['relevance_score'] * (1-r['agree_score'])
                #print(parsed_titles[i],r['agree_score'],r['relevance_score'],r['ranking_score'])
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
        
    return render_template('search.html', name=project_name, net_id=net_id, output_message=output_message, data=result)
