from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn

class TfidfEmbedding(nn.Module): #just Tfidf with different name
    def __init__(self):
        super(TfidfEmbedding, self).__init__()
        self.model = TfidfVectorizer()
    def embed(self, text):
        embedding = self.model.fit_transform(text)
        return embedding.toarray()



