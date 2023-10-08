import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class Linear(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.lr = kwargs.get('lr')

        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(128, 10)

        self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('Training loss', loss.item())
        return loss

    def on_validation_start(self):
        self.losses = []
        self.accuracies = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = F.cross_entropy(probs, y)

        acc = self.accuracy(probs, y)
        self.accuracies.extend(acc.cpu().numpy().tolist())
        self.losses.append(loss.item())
        return loss

    def validation_epoch_end(self, outputs):
        overall_acc = np.mean(self.accuracies)
        overall_loss = np.mean(self.losses)
        self.log('Validation loss', overall_loss)
        self.log('Validation Accuracy', overall_acc)

    def on_test_start(self):
        self.accuracies = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.accuracies.extend(acc.cpu().numpy().tolist())
        return acc

    def test_epoch_end(self, outputs):
        overall_acc = np.mean(self.accuracies)
        self.log("Test Accuracy", overall_acc)

    def accuracy(self, logits, y):
        acc = torch.eq(torch.argmax(logits, -1), y).to(torch.float32)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)