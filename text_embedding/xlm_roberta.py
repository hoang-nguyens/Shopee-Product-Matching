from transformers import XLMTokenizer, XLMModel
import torch
import torch.nn as nn

class XLMRoberta(nn.Module):
    def __init__(self, model_name, fc_dim = 512):
        super(XLMRoberta, self).__init__()
        self.tokenizer = XLMTokenizer.from_pretrained(model_name)
        self.model = XLMModel.from_pretrained(model_name)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)


        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    def embed(self, text):
        input = self.tokenizer(text,
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=16)
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']

        device = next(self.parameters()).device  # Get device from model parameters
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        embedding = outputs.last_hidden_state[:,0,:]

        return embedding

    def forward(self, text):
        out = self.embed(text)
        out = self.fc(out)
        out = self.bn(out)
        return out




