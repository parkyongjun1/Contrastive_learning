import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils import data
import torchmetrics
from torchvision.datasets.stl10 import STL10
from models.backbone import ResNetSimCLR 
from models.simclr import SimCLR
from datasets.loader import STL10DataModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

class PretrainedModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

class Transfer(pl.LightningModule):
        def __init__(self, model, lr, weight_decay, len):
            super().__init__()
            self.model = model
            self.model.model.head = nn.Sequential(
                #nn.Linear(self.model.model.head[-1].in_features, self.model.model.head[-1].in_features),
                #nn.ReLU(),
                nn.Linear(self.model.model.head[-1].in_features, 10)
                )
            
            self.lr = lr
            self.decay = weight_decay
            self.len = len
            
            self.criterion = nn.CrossEntropyLoss()
            self.metric = torchmetrics.Accuracy()
            
        def forward(self, x):
            x = self.model(x)
            return x
        
        def training_step(self, batch, batch_idx):    
            x, target = batch
            pred = self(x)
            loss = self.criterion(pred, target)
            self.log('train_loss_step',loss)
            
            return {"loss": loss, "pred": pred, "target": target}
        
        def training_epoch_end(self, outputs):
            avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean(dtype=torch.float)
            pred = torch.stack([x['pred'] for x in outputs])
            target = torch.stack([x['target'] for x in outputs])
            
            self.log("Avg_train_loss_Epoch", avg_train_loss)
            
            accuracy = self.metric(pred.argmax(dim=-1), target)
            self.log('train_acc', accuracy)
             
            

        def validation_step(self, batch, batch_idx):    
            x, target = batch
            pred = self(x)        
            loss = self.criterion(pred, target)
            self.log('val_loss_step',loss)
            
            return {"loss": loss, "pred": pred, "target": target}
        
        def validation_epoch_end(self, outputs):
            avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
            pred = torch.stack([x['pred'] for x in outputs])
            target = torch.stack([x['target'] for x in outputs])
            self.log("Avg_val_loss_Epoch", avg_train_loss)

            accuracy = self.metric(pred.argmax(dim=-1), target)
            self.log('val_acc', accuracy)
        
        def configure_optimizers(self):
            optimizer=  torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.len, eta_min=0, last_epoch=-1)
            return [optimizer], [scheduler]
        

if __name__=="__main__":
    batch_size = 125
    
    extractor = ResNetSimCLR(base_model='resnet18', out_dim=128)
    
    pretrained_model = PretrainedModel.load_from_checkpoint(checkpoint_path='/home/ssb/research/pretrain_r18_train/epoch=105-Avg_val_accuracy=0.95563.ckpt',model=extractor, strict=False)
    pretrained_model.freeze()

    dataset = STL10DataModule(data_dir='/home/ssb/research/dataset/', batch_size=batch_size)
    stl10_all, stl10_unlabeled, stl10_val, stl10_train, stl10_test = dataset.setup()
    trainLoader = DataLoader(stl10_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    valLoader = DataLoader(stl10_test, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    #model = Transfer(model=extractor, lr=0.0003, weight_decay=0.0008, len=len(trainLoader))
    model = Transfer(model=pretrained_model, lr=0.0003, weight_decay=0.0008, len=len(trainLoader))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath= 'finetuning_r18_unlabeled',
        filename= '{epoch}-{val_acc:.5f}',
        verbose=True,
        save_last=True,
        save_top_k=5,
        monitor='val_acc',
        mode='max',
        save_weights_only=True,
        )
    
    trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=300, gpus=4, strategy="ddp", precision=16)    
    trainer.fit(model, trainLoader, valLoader)