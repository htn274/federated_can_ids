import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent import futures
from functools import partial
import tqdm 

np.random.seed(0)

class Writer:
    def __init__(self, outdir, start_idx=0, is_train=True):
        self.outdir = Path(outdir)/ ('train' if is_train else 'val')
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idx = start_idx
    def write(self, X, y):
        save_file = self.outdir / f'{self.idx}.npz'
        np.savez_compressed(save_file, X=X, y=y)
        self.idx += 1
    def start(self):
        print('Start writing to: ', self.outdir)

def write_to_file(writer, X, y):
    writer.start()
    try:
        for xi, yi in tqdm.tqdm(zip(X, y)):
            writer.write(xi, yi)
    except: return False
    return True

def train_test_split(N, test_fraction=0.3):
    test_size = int(N * test_fraction)
    indices = np.random.permutation(N) 
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return train_idx, test_idx
    
def process_dataset(X, y, train_writer, val_writer, test_fraction=0.3):
    train_idx, test_idx = train_test_split(y.shape[0], test_fraction)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    write_to_file(train_writer, X_train, y_train)
    write_to_file(val_writer, X_test, y_test)
    return train_idx, test_idx 

def main(indir, outdir, attacks, split_id=1, test_fraction=0.3):
    outdir = Path(outdir) / str(split_id)
    train_writer = Writer(outdir=outdir)
    val_writer = Writer(outdir=outdir, is_train=False)
    normals = ['Normal_' + x for x in attacks]
    files = attacks + normals
    for a in files:
        filename = f'{a}.npz'
        filename = Path(indir) / filename
        print('Processing: ', filename)
        data = np.load(filename)
        X, y = data['X'], data['y']
        y = y.squeeze()
        train_idx, test_idx = process_dataset(X, y, train_writer, val_writer, test_fraction)
        print(f'Train size: {len(train_idx)}, Test size: {len(test_idx)}')
        np.savez_compressed(outdir / 'idex.npz', train=train_idx, test=test_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--id', type=int)
    parser.add_argument('--f', type=float)
    args = parser.parse_args()
    # indir = '../Data/CHD_w29_s14_ID_Data/wavelet/gaus1/'
    # outdir = '../Data/CHD_w29_s14_ID_Data/wavelet/gaus1/'
    # attack_list = ['DoS', 'Fuzzy', 'gear', 'RPM']
    attack_list = ['Normal', 'Fuzzy', 'Replay']
    if args.outdir is None:
        args.outdir = args.indir
    main(args.indir, args.outdir, attack_list, split_id=args.id, test_fraction=args.f)
