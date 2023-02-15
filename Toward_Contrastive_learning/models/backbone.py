import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super().__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet34": models.resnet34(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        self.backbone = self._get_basemodel(base_model)
        in_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,out_dim)
        )

    def _get_basemodel(self, model_name):
            model = self.resnet_dict[model_name]
            return model

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.head(x)
        return x
