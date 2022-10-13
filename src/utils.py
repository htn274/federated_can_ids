from typing import OrderedDict
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import torch

def get_parameters(model):
    """
    Return a list of parameters of a model
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, params):
    params_dicts = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dicts})
    model.load_state_dict(state_dict, strict=True)


def cal_metric(label, pred):
    cm = confusion_matrix(label, pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2*recall*precision / (recall + precision)
    
    total_actual = np.sum(cm, axis=1)
    true_predicted = np.diag(cm)
    fnr = (total_actual - true_predicted)*100/total_actual
                   
    return cm, {
    'fnr': np.array(fnr),
    'rec': recall,
    'pre': precision,
    'f1': f1
    }

def print_results(results, classes):
    print('\t' + '\t'.join(map(str, results.keys())))
    for idx, c in enumerate(classes):
        res = [round(results[k][idx], 4) for k in results.keys()]
        output = [c] + res
        print('\t'.join(map(str, output)))