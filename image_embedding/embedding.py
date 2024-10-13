# efficentnet, vit, nfnet
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision.models as models
import transformers


def gem(x, p =3, eps = 1e-6): # generalize average pooling # use for model that return features in 2d instead of 1d
    return F.avg_pool2d(x.clamp(min = eps).pow(p), kernel_size=(x.shape[-2], x.shape[-1])).pow(1/p)

class EmbeddingNet(nn.Module):
    def __init__(self, model_name, fc_dim = 512):
        super(EmbeddingNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained= True)
        if 'vit' in model_name or 'nfnet' in model_name :
            self.out_features = self.backbone.head.out_features
        if 'efficient' in model_name:
            self.out_features = self.backbone.classifier.out_features
        self.fc = nn.Linear(self.out_features, fc_dim) # using efficientnet train with ImageNet dataset, so num_features is 1000
        self.bn = nn.BatchNorm1d(fc_dim)

    def _init_params(self):
        nn.init.xavier_normal(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        out = self.bn(out)
        return out

