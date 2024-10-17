import math

import torch.nn as nn
import torch
import torch.nn.functional as F

class Adacos(nn.Module):
    def __init__(self, output, input = 512):
        super(Adacos, self).__init__()
        self.input = input
        self.output = output

        self.W = nn.Parameter(torch.FloatTensor(output, input))
        nn.init.xavier_normal_(self.W)

    def forward(self, embed_vector):
        embedding = F.normalize(embed_vector, dim = 1, p = 2)
        weight = F.normalize(self.W, dim = 1, p = 2)

        cos = torch.matmul(embedding, weight.T)
        sin = torch.sqrt(1 - cos ** 2)
        sin = sin.clamp(1e-6, 1)

        cos_mean = torch.mean(cos, dim=0)
        sin_mean = torch.mean(sin, dim =0)
        theta = torch.atan2(sin_mean, cos_mean)

        B = embed_vector.shape[0]
        s = math.log( (B - 1) / math.pi) * torch.cos(theta) * math.pi
        logit = cos * s
        return F.softmax(logit, dim = 1)

