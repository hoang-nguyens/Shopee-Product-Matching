import torch.nn as nn
import torch
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, output, input = 512 , m = 0.5, s = 30.0): # mode is image or text
        super(ArcFace, self).__init__()
        self.m = m
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.s = s
        self.input = input
        self.output = output

        self.W = nn.Parameter(torch.FloatTensor(output, input))
        nn.init.xavier_normal_(self.W)

    def forward(self, embed_vector, label = None):
        embedding = F.normalize(embed_vector, dim = 1, p = 2)
        weight = F.normalize(self.W, dim = 1, p = 2)

        cos = torch.matmul(embedding, weight.T) # take norm above so we don't need to divide by norm anymore
        if label is None:
            logit = cos * self.s
        else: # cos(x + m) = cosx. cosm - sinx.sinm
            sin = torch.sqrt(1 - cos ** 2)
            sin = sin.clamp(1e-7, 1)
            cos_theta = cos * self.cos_m - sin * self.sin_m

            onehot = torch.zeros_like(cos)

            onehot.scatter_(dim = 1,index = label.view(-1, 1), value= 1) # make sure that index is tensor

            logit = cos * ( 1 - onehot) + cos_theta * onehot
            logit = logit * self.s
        return F.softmax(logit, dim = 1)







