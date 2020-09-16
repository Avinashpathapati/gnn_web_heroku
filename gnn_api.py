
import sys
import os
import random
from ase.io import read
from schnetpack import AtomsData
import schnetpack as spk
from tqdm import tqdm

from flask import Blueprint, request, jsonify, Flask
import torch
import torch.nn.functional as F
app = Flask(__name__)
api = Blueprint('api', __name__)


def gnn_pred(cif_file):

    # device = torch.device("cuda" if args.cuda else "cpu")
    device = "cpu"
    sch_model = torch.load(os.path.join("./schnetpack/model", 'best_model'), map_location=torch.device(device))
    test_dataset = AtomsData('./cod_predict.db')
    test_loader = spk.AtomsLoader(test_dataset, batch_size=32)
    prediction_list = []
    for count, batch in enumerate(test_loader):
            
            # move batch to GPU, if necessary
            print('before batch')
            batch = {k: v.to(device) for k, v in batch.items()}
            print('after batch')
            # apply model
            pred = sch_model(batch)
            prediction_list.extend(pred['band_gap'].detach().cpu().numpy().flatten().tolist())
    
    return prediction_list[0]


@api.route('/predict', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    review's text data.
    '''
    if request.method == 'POST':
        if 'cif_data' not in request.form:
            return jsonify({'error': 'no review in body'}), 400
        else:
            print('in rest api')
            cif_file = request.form['file_name']
            cif_data = request.form['cif_data']
            output = gnn_pred(cif_file)
            print(output)
            return jsonify(float(output))


app.register_blueprint(api, url_prefix='/api')
if __name__ == '__main__':
    app.run(debug=True)