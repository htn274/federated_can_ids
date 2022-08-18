import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from asyncio import as_completed
from concurrent import futures

class Writer:
    def __init__(self, outdir, start_idx=0):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idx = start_idx
    def write(self, X, y):
        save_file = self.outdir / f'{self.idx}.pt'
        torch.save((X, y), save_file)
        self.idx += 1
        # print(self.idx)

def write_to_file(writer, X, y):
    try:
        for xi, yi in zip(X, y):
            writer.write(xi, yi)
    except: return False
    return True

def process_dataset(sss, X, y, train_writers, val_writers, num_splits):
    with futures.ThreadPoolExecutor(num_splits) as exec:
        todo = []
        for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            print(i, len(train_idx), len(test_idx))
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[test_idx]
            y_val = y[test_idx]
            future = exec.submit(write_to_file, train_writers[i], X_train, y_train)
            todo.append(future)
            future = exec.submit(write_to_file, val_writers[i], X_val, y_val)
            todo.append(future)

        sucess = 0
        fail = 0
        for future in futures.as_completed(todo):
            res = future.result()
            if res:
                sucess += 1
            else:
                fail += 1
    return sucess, fail

def main(indir, outdir, attacks, num_splits=5, test_size=0.3):
    train_writers = [Writer(outdir=outdir + f'train/{i + 1}/') for i in range(num_splits)]
    val_writers = [Writer(outdir=outdir + f'val/{i + 1}/') for i in range(num_splits)]
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
    for a in attacks:
        filename = f'{a}.pt'
        filename = Path(indir) / filename
        print('Processing: ', filename)
        X, y = torch.load(filename)
        sss.get_n_splits(X, y)
        sucess, fail = process_dataset(sss, X, y, train_writers, val_writers, num_splits)
        print(f'Sucess: {sucess} - Fails: {fail}')
        # print(train_writers[0].idx)

if __name__ == '__main__':
    indir = '../Data/CHD_w29_s14_ID_Data/wavelet/'
    outdir = '../Data/CHD_w29_s14_ID_Data/'
    attack_list = ['DoS', 'Fuzzy', 'gear', 'RPM']
    main(indir, outdir, attack_list, num_splits=5, test_size=0.3)