from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Changing-Views"
net_id = "Yuji Akimoto (ya242) , Benjamin Edwards (bje43) , Jacqueline Wen (jzw22) , Young Kim (yk465) , Zachary Brienza (zb43)"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



