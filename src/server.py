import pickle
from typing import OrderedDict
import flwr as fl
import torch
from central_ids import IDS
from custom_strategy import SaveModelStrategy
from pathlib import Path
import numpy as np
from dataset import CANDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

DEVICE = "cuda"
DEFAULT_SERVER_ADDRESS = "[::]:8080"

def load_data(car_models, data_dir):
    transform = None
    data_loaders = []
    for car_model in car_models:
        test_dataset = CANDataset(root_dir=Path(data_dir.format(car_model))/'test', is_binary=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, 
                                pin_memory=True, sampler=None)
        data_loaders.append(test_loader)
    return data_loaders

def evaluate(model, test_loaders:list):
    # res = []
    total_labels = np.array([])
    total_preds = np.array([])
    for test_loader in test_loaders:
        trainer = pl.Trainer(enable_checkpointing=False, logger=False)
        results = trainer.predict(model, dataloaders=test_loader)
        labels = np.concatenate([x['labels'] for x in results])
        preds = np.concatenate([x['preds'] for x in results])
        f1 = f1_score(labels, preds)
        # res.append(f1)
        total_labels = np.concatenate([total_labels, labels])
        total_preds = np.concatenate([total_preds, preds])
    f1_global = f1_score(total_labels, total_preds)
    # res.append(f1_global)
    return f1_global

def get_evaluate_fn():
    car_models = ['Kia', 'Tesla', 'BMW'] 
    data_dir = '../../Data/LISA/{}/1/'
    test_loaders = load_data(car_models=car_models, data_dir=data_dir)
    kwargs = {'C': 2}
    model = IDS(**kwargs)
    def centralized_eval( server_round: int,
        parameters: fl.common.NDArrays,
        config,):
        param_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:  torch.tensor(np.atleast_1d(v)) for k, v in param_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        f1_global = evaluate(model, test_loaders)
        metrics = {'f1_global': f1_global}
        return 0.0, metrics
    return centralized_eval

def save_hist(history, save_dir):
    f = open(Path(save_dir) / "hist.pkl", "wb")
    pickle.dump(history, f)
    f.close()

if __name__ == '__main__':
    save_dir = '../save/federated/'
    strategy = SaveModelStrategy(
        fraction_fit = 1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn = get_evaluate_fn(),
        # on_fit_config_fn=fit_config,
        save_dir=Path(save_dir),
    )
    hist = fl.server.start_server(
        server_address=DEFAULT_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
    print("Saving training history")
    save_hist(hist, save_dir)