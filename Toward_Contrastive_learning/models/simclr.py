import torch
import torch.nn as nn
import torch.nn.functional as F

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SimCLR(object):
    def __init__(self, model, tau):
        super().__init__()
        self.model = model
        self.tau = tau
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def nce_loss(self, features1, features2):
        
        features1 = F.normalize(features1)
        features2 = F.normalize(features2)
        
        features1_gather = concat_all_gather(features1)
        features2_gather = concat_all_gather(features2)
        
        logits12 = (features1 @ features2_gather.T) / self.tau
        logits21 = (features2 @ features1_gather.T) / self.tau
        
        bs = logits12.shape[0]

        labels12 = (torch.arange(bs, dtype=torch.long) + bs * torch.distributed.get_rank()).cuda()
        labels21 = (torch.arange(bs, dtype=torch.long) + bs * torch.distributed.get_rank()).cuda()
        
        loss = 0.5 * (self.criterion(logits12, labels12) + self.criterion(logits21, labels21))
        
        return loss, logits12, labels12, logits21, labels21

    def forward(self, x):
        x1, x2 = x
        
        features1 = self.model(x1)
        features2 = self.model(x2)
        
        loss, out12, out21 = self.nce_loss(features1, features2)
        
        return loss, out12, out21
