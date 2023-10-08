import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class Conv(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.lr = kwargs.get('lr')

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output


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