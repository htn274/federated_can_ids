from typing import OrderedDict
from anyio import Path
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import CANDataset

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

def test_model(data_dir, model):
    transform = None
    test_dataset = CANDataset(root_dir=Path(data_dir)/'test', is_binary=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, 
                        pin_memory=True, sampler=None)
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    results = trainer.predict(model, dataloaders=test_loader)
    labels = np.concatenate([x['labels'] for x in results])
    preds = np.concatenate([x['preds'] for x in results])
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    f1 = f1_score(labels, preds)
    err = (fn + fp) / len(preds)
    return (f1, err)