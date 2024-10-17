import torch
from image_embedding import EmbeddingNet
from text_embedding import BertEmbedding
from data import ShopeeDataset, getDataloader, getTransform
from similarity import ArcFace
from grouping import KNNPredict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm


def approach_1():
    df = pd.read_csv('train.csv')
    IMG_DIR = 'train_images/'

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label_group'].nunique()

    dataloader = getDataloader(df, IMG_DIR, 32, True, getTransform(), True)

    image_embedding_model_name = 'efficientnet_b0'
    image_embedding_model = EmbeddingNet(image_embedding_model_name, 1792)
    image_arcface = ArcFace(num_classes, 1792)

    text_embedding_model_name = 'bert-base-uncased'
    text_embedding_model = BertEmbedding(text_embedding_model_name, 1024)
    text_arcface = ArcFace(num_classes, 1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_embedding_model.to(device)
    image_arcface.to(device)
    text_embedding_model.to(device)
    text_arcface.to(device)

    image_embedd_final = []
    text_embedd_final = []

    for image, title, label in tqdm(dataloader):
        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            image_embedd = image_embedding_model(image)
            image_embedd = image_arcface(image_embedd, label).detach().to('cpu').numpy()
            image_embedd_final.append(image_embedd)

            text_embedd = text_embedding_model(title)
            text_embedd = text_arcface(text_embedd, label).detach().to('cpu').numpy()
            text_embedd_final.append(text_embedd)

    image_embedd_final = np.vstack(image_embedd_final)
    text_embedd_final = np.vstack(text_embedd_final)

    predict_model_image = KNNPredict(df, image_embedd_final)
    predict_model_text = KNNPredict(df, text_embedd_final)

    prediction_image = predict_model_image.predict()
    prediction_text = predict_model_text.predict()

    return prediction_image, prediction_text

if __name__ == '__main__':
    out1, out2 = approach_1()
