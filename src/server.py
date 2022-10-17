import pickle
import flwr as fl
from custom_strategy import SaveModelStrategy
from pathlib import Path

DEVICE = "mps"
DEFAULT_SERVER_ADDRESS = "[::]:8080"

def save_hist(history, save_dir):
    f = open(Path(save_dir) / "hist.pkl", "wb")
    pickle.dump(history, f)
    f.close()

if __name__ == '__main__':
    save_dir = '../save/federated/'
    strategy = SaveModelStrategy(
        fraction_fit = 1.0,
        fraction_eval=1.0,
        min_fit_clients=3,
        # min_eval_clients=3,
        min_available_clients=3,
        # eval_fn = get_eval_fn(),
        # on_fit_config_fn=fit_config,
        save_dir=Path(save_dir),
    )
    hist = fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        config={"num_rounds": 10},
        strategy=strategy
    )
    print("Saving training history")
    save_hist(hist, save_dir)