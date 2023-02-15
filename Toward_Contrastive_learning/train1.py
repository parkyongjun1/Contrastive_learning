import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils import data
import torchmetrics
from pl_bolts.optimizers.lars import LARS
from torchvision.datasets.stl10 import STL10
from models.simclr import SimCLR
from datasets.loader import STL10DataModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


class BaseTrainer(pl.LightningModule):
        def __init__(self, base_encoder, out_dim, tau, lr, weight_decay, momentum, length):
            super().__init__()
            self.model = SimCLR(base_encoder=base_encoder, out_dim=out_dim, tau=tau)       
            self.criterion = nn.CrossEntropyLoss()
            
            self.lr = lr
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.length = length

            self.train_metric = torchmetrics.Accuracy()            
            self.val_metric = torchmetrics.Accuracy() 
            self.save_hyperparameters("base_encoder", "out_dim", "tau", "lr", "weight_decay", "momentum")
            
        def forward(self, x):
            img, target = x
            
            logits, labels = self.model(img)

            return logits, labels
        
        def training_step(self, batch, batch_idx):    
            logits, labels = self(batch)
            
            loss = self.criterion(logits, labels)

            
            positive_min = logits[:,0].detach().min()
            negative_max = logits[:,1:].detach().max()
            positive_mean = logits[:,0].detach().mean()
            negative_mean = logits[:,1:].detach().mean()
        
            
            acc = self.train_metric(logits.softmax(dim=-1),labels)
            return {"loss": loss, "acc": acc,
                    "positive_min": positive_min, "negative_max": negative_max, 
                    "positive_mean": positive_mean, "negative_mean": negative_mean}
        
        def training_epoch_end(self, outputs):
            avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
            positive_min = torch.stack([x['positive_min'] for x in outputs]).min()
            negative_max  = torch.stack([x['negative_max'] for x in outputs]).max()
            positive_mean = torch.stack([x['positive_mean'] for x in outputs]).mean()
            negative_mean = torch.stack([x['negative_mean'] for x in outputs]).mean()
            
            self.log("(Train)Loss(Avg)", avg_train_loss)
            self.log("(Train)Positive(min)", positive_min)
            self.log("(Train)Negative(max)", negative_max)
            self.log("(Train)Positive(mean)", positive_mean)
            self.log("(Train)Negative(mean)", negative_mean)
            
            top1 = self.train_metric.compute()
            self.log('Top1_Accuracy_Train', top1)
            self.train_metric.reset()
            
            

        def validation_step(self, batch, batch_idx):    
            logits, labels = self(batch)
            
            loss = self.criterion(logits, labels)

            
            positive_min = logits[:,0].detach().min()
            negative_max = logits[:,1:].detach().max()
            positive_mean = logits[:,0].detach().mean()
            negative_mean = logits[:,1:].detach().mean()
        
            
            acc = self.val_metric(logits.softmax(dim=-1),labels)
            return {"loss": loss, "acc": acc,
                    "positive_min": positive_min, "negative_max": negative_max, 
                    "positive_mean": positive_mean, "negative_mean": negative_mean}
        
        def validation_epoch_end(self, outputs):
            avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
            positive_min = torch.stack([x['positive_min'] for x in outputs]).min()
            negative_max  = torch.stack([x['negative_max'] for x in outputs]).max()
            positive_mean = torch.stack([x['positive_mean'] for x in outputs]).mean()
            negative_mean = torch.stack([x['negative_mean'] for x in outputs]).mean()
            
            self.log("(Val)Loss(Avg)", avg_val_loss)
            self.log("(Val)Positive(min)", positive_min)
            self.log("(Val)Negative(max)", negative_max)
            self.log("(Val)Positive(mean)", positive_mean)
            self.log("(Val)Negative(mean)", negative_mean)
            
            top1 = self.val_metric.compute()
            self.log('Top1_Accuracy_Val', top1)
            self.val_metric.reset()
        
        def configure_optimizers(self):
            optimizer = LARS(self.parameters(), self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.length, eta_min=0, last_epoch=-1)
            #return [optimizer], [scheduler]
            return optimizer

if __name__ == "__main__":
    
    batch_size =2048
    
    STL10Dataset = STL10DataModule(data_dir='/home/ssb/research/dataset/', batch_size=batch_size)
    STL10_unlabeled, STL10_val = STL10Dataset.setup()
    
    TrainLoader = DataLoader(STL10_unlabeled, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    ValLoader = DataLoader(STL10_val, batch_size=512, shuffle=False, num_workers=8, drop_last=True)
    
    model = BaseTrainer(base_encoder='resnet18', out_dim=128, tau=0.07, lr=0.6, weight_decay=0.0001, momentum=0.9, length=len(TrainLoader))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath= 'pretrained_r18_unlabeled',
        filename= '{epoch}-{Top1_Accuracy_Val:.5f}',
        verbose=True,
        save_last=True,
        save_top_k=5,
        monitor='Top1_Accuracy_Val',
        mode='max',
        save_weights_only=True,
        )
    trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=200, gpus=4, strategy="ddp", precision=16) 
    trainer.fit(model, TrainLoader, ValLoader)
