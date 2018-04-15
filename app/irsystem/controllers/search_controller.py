from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
import sys
sys.path.insert(1, '../../../analysis')
from word_embeddings import GloVe
from tfidf_query import find_keywords, topic_search

project_name = "Changing-Views"
net_id = "Yuji Akimoto (ya242), Benjamin Edwards (bje43), Jacqueline Wen (jzw22), Young Kim (yk465), Zachary Brienza (zb43)"

with open('../data/sample.json', 'r') as f:
    data = json.load(f)

data = find_keywords(data, n=10)
glove = GloVe('../../../data/glove.6B.zip')


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    if not query:
        result = []
        output_message = ''
    else:
        output_message = 'Your search: ' + query
        result = topic_search(query, data, glove)
    return render_template('search.html', name=project_name, net_id=net_id, output_message=output_message, data=result)
