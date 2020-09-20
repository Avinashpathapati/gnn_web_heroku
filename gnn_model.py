
import sys
import os
import random
from ase.io import read
from schnetpack import AtomsData
import schnetpack as spk
from tqdm import tqdm

import torch
import torch.nn.functional as F



def gnn_pred():

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