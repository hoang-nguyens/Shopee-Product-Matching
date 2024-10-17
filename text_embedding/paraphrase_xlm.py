import torch
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn as nn

class ParaphraseEmbedding(nn.Module):
    def __init__(self, model_name: str, fc_dim: int = 512):
        super(ParaphraseEmbedding, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def embed(self, text: str):
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            max_length=16,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Move tensors to the correct device if using GPU
        device = next(self.parameters()).device  # Get device from model parameters
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        print(input_ids.device)
        print(attention_mask.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute the mean of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1)
        print(embedding.device)

        return embedding


    def forward(self, text: str):
        out = self.embed(text)
        out = self.fc(out)
        out = self.bn(out)
        print(out.device)
        return out
