import json
import sys

from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from analysis.word_embeddings import GloVe
from analysis.tfidf_query import find_keywords, topic_search
from . import *

project_name = "Changing-Views"
net_id = "Yuji Akimoto (ya242), Benjamin Edwards (bje43), Jacqueline Wen (jzw22), Young Kim (yk465), Zachary Brienza (zb43)"

with open('./data/data.json', 'r') as f:
    data = json.load(f)

data, dt_matrix, vocab = find_keywords(data, n=10)
glove = GloVe('./data/glove.6B.50d.txt')


@irsystem.route('/', methods=['GET'])
def home():
  print("wtf")
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
  print("wtfx2")
  query = request.args.get('search')
  if not query:
    result = []
    output_message = ''
  else:
    output_message = 'Your search: ' + query
    result = topic_search(query, data, glove, dt_matrix, vocab)
  return render_template('search.html', name=project_name, net_id=net_id, output_message=output_message, data=result)
