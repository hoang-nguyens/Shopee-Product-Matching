import torch
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn as nn

class ParaphraseEmbedding(nn.Module):
    def __init__(self, model_name):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=16,
                                truncation=True)
        inputs_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(inputs_ids = inputs_ids, attention_mask = attention_mask)

        embedding = outputs.last_hidden_state.mean(dim = 1)
        return embedding
