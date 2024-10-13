from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, model_name):
        super(BertEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        input = self.tokenizer(text,
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=16)
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        embedding = outputs.last_hidden_state[:,0,:]
        return embedding



