import torch
import torch.nn as nn
import torch.nn.functional as F

class CurricularFace(nn.Module):
    def __init__(self, output, input = 512, m = 0.5, s = 30.0, lambda_max = 5, lambda_0 = 0.01, alpha = 0.001 ):
        super(CurricularFace, self).__init__()
        self.output = output
        self.input = input
        self.m = m
        self.s = s
        self.lambda_max = lambda_max
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))

        self.W = nn.Parameter(torch.FloatTensor(output, input))
        nn.init.xavier_normal(self.W)



    def forward(self, embedded_image, label = None, t = 0):
        embedd = F.normalize(embedded_image, p = 2, dim = 1)
        weight = F.normalize(self.W, p = 2, dim = 1)

        cos_theta = torch.matmul(embedd, weight.T)

        if label is None:
            logit = cos_theta * self.s
        else:
            sin_theta = torch.sqrt(1 - cos_theta ** 2)
            sin_theta = sin_theta.clamp(1e-6, 1)
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

            mask = cos_theta > cos_theta_m
            cos_theta_m = torch.where(mask, cos_theta_m, cos_theta_m * max(self.lambda_max, self.lambda_0 + t * self.alpha))

            onehot = torch.zeros_like(cos_theta_m)
            onehot.scatter_(1, label.view(-1, 1), 1.0)

            logit = (1 - onehot) * cos_theta + onehot * cos_theta_m

        return F.softmax(logit)


