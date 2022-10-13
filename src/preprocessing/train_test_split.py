import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent import futures
from functools import partial
import tqdm 

np.random.seed(0)

class Writer:
    def __init__(self, outdir, type_name, start_idx=0,):
        self.outdir = Path(outdir)/ type_name
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

def train_test_split(N, val_fraction, test_fraction):
    test_size = int(N * test_fraction)
    val_size = int(N * val_fraction)
    indices = np.random.permutation(N) 
    val_idx = indices[:val_size]
    test_idx = indices[val_size:val_size + test_size]
    train_idx = indices[val_size + test_size:]
    return [train_idx, val_idx, test_idx]
    
def process_dataset(X, y, writer_list, val_fraction, test_fraction, shuffle):
    if shuffle:
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
    train_val_test_idx = train_test_split(len(y), val_fraction, test_fraction)
    title = ['Train', 'Val', 'Test']
    for (name, idx, writer) in zip(title, train_val_test_idx, writer_list):
        print(f"{name}: {len(idx)}")
        X_subset, y_subset = X[idx], y[idx]
        write_to_file(writer, X_subset, y_subset)
    return train_val_test_idx

def main(indir, outdir, attacks, split_id, val_fraction, test_fraction, shuffle=True):
    outdir = Path(outdir) / str(split_id)
    train_writer = Writer(outdir=outdir, type_name='train')
    val_writer = Writer(outdir=outdir, type_name='val')
    test_writer = Writer(outdir=outdir, type_name='test')
    writer_list = [train_writer, val_writer, test_writer]
    # normals = ['Normal_' + x for x in attacks]
    files = attacks #+ normals
    for a in files:
        filename = f'{a}.npz'
        filename = Path(indir) / filename
        print('Processing: ', filename)
        data = np.load(filename)
        X, y = data['X'], data['y']
        y = y.squeeze()
        _ = process_dataset(X, y, writer_list, val_fraction, test_fraction, shuffle)
        # print(f'Train size: {len(train_idx)}, Test size: {len(test_idx)}')
        # np.savez_compressed(outdir / 'idex.npz', train=train_idx, test=test_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--id', type=int)
    parser.add_argument('--test_fraction', type=float, default=0.2)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    args = parser.parse_args()
    # indir = '../Data/CHD_w29_s14_ID_Data/wavelet/gaus1/'
    # outdir = '../Data/CHD_w29_s14_ID_Data/wavelet/gaus1/'
    # attack_list = ['DoS', 'Fuzzy', 'gear', 'RPM']
    attack_list = ['Normal', 'Fuzzy', 'Replay']
    if args.outdir is None:
        args.outdir = args.indir
    main(args.indir, args.outdir, attack_list, split_id=args.id, 
        val_fraction=args.val_fraction, test_fraction=args.test_fraction)
