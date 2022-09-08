from curses.ascii import CAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import f1_score

from networks.resnet import resnet18
from dataset import CANDataset
from torch.utils.data import DataLoader

class Classifier(pl.LightningModule):
    def __init__(self, num_classes, data_dir) -> None:
        super(Classifier, self).__init__()
        self.encoder = resnet18()
        self.fc = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.data_dir = data_dir

    def forward(self, x):
        return self.fc(self.encoder(x))

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
        # print(preds)
        return {'preds': preds, 'labels': y.numpy()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        f1 = f1_score(logits.cpu(), y.cpu(), num_classes=3, top_k=1)
        self.log('val_f1', f1, logger=True)
        self.log('val_loss', avg_loss, logger=True)
        print(f'Validation f1 score: {f1:.4f}')
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                          lr=5e-4,
                          momentum=0.9,
                          weight_decay=1e-4)
        return optimizer  

    def prepare_data(self):
        transform = None
        self.train_dataset = CANDataset(root_dir=self.data_dir, is_train=True, transform=transform)
        self.val_dataset = CANDataset(root_dir=self.data_dir, is_train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True, 
                        pin_memory=True, sampler=None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, shuffle=False,
                        pin_memory=True, sampler=None)
    
if __name__ == '__main__':
    data_dir = '../../Data/LISA/Federated_Data/Preprocessed_Data/Kia/1/'
    logger = TensorBoardLogger(save_dir='../save/',) 
    model = Classifier(data_dir=data_dir, num_classes=3)
    trainer = pl.Trainer(max_epochs=50, accelerator='mps', 
                        logger=logger, log_every_n_steps=100, check_val_every_n_epoch=5)
    trainer.fit(model)