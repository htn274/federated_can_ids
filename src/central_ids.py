from argparse import ArgumentParser
from gc import callbacks
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import f1_score

from dataset import CANDataset
from torch.utils.data import DataLoader
from networks.classifier import Classifier


class IDS(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super(IDS, self).__init__()
        self.save_hyperparameters()
        self.args = kwargs
        self.model = Classifier(num_classes=self.args['C'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        outs = self.forward(x)
        _, preds = outs.topk(1, 1, True, True)
        preds = preds.t().cpu().numpy().squeeze(0)
        return preds

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, logger=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        return {'val_loss': loss, 'labels': y, 'logits': logits}

    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.predict(X)
        return {'preds': preds, 'labels': y.numpy()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        f1 = f1_score(logits.cpu(), y.cpu(), num_classes=self.args['C'], top_k=1)
        self.log('val_f1', f1, logger=True)
        self.log('val_loss', avg_loss, logger=True)
        print(f'Validation f1 score: {f1:.4f}')
    
    def configure_optimizers(self):
        self.optimizer = optim.SGD(self.parameters(),
                          lr=self.args['lr'],
                          momentum=0.9,
                          weight_decay=self.args['weight_decay'])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6, verbose=True,
            ),
            # "interval": "epoch",
            "monitor": "train_loss",
        }
        return [self.optimizer], [self.scheduler]

    def prepare_data(self):
        transform = None
        binary = False
        if self.args['C'] == 2:
            binary = True
        self.train_dataset = CANDataset(root_dir=Path(self.args['data_dir']) / 'train', 
                                        is_binary=binary,
                                        transform=transform)
        self.val_dataset = CANDataset(root_dir=Path(self.args['data_dir']) / 'val', 
                                        is_binary=binary, 
                                        transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args['B'], shuffle=True, 
                        pin_memory=True, sampler=None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=512, shuffle=False,
                        pin_memory=True, sampler=None)

    def get_train_size(self):
        return len(self.train_dataset)

    def get_val_size(self):
        return len(self.val_dataset)

def argument_paser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

def main():
    """
    Centralized training.
    """
    args = argument_paser()
    logger = loggers.TensorBoardLogger(save_dir=args['save_dir'],) 
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.2f}-{val_f1:.4f}',
        every_n_epochs=args['val_freq'],
        monitor='val_loss', 
        save_top_k=5)
    model = IDS(**args)
    trainer = pl.Trainer(max_epochs=args['epochs'], accelerator='mps', 
                        logger=logger, log_every_n_steps=100, 
                        check_val_every_n_epoch=args['val_freq'],
                        callbacks=[checkpoint_callback])
    trainer.fit(model)

if __name__ == '__main__':
    main()    