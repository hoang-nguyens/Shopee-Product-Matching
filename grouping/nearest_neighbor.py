from sklearn.neighbors import NearestNeighbors
import numpy as np


class KNNPredict:
    def __init__(self, df, embedd, num_neighbors = 50): # data here is collector of embedded vectors
        self.df = df
        self.embedd = embedd
        self.knn = NearestNeighbors(n_neighbors=num_neighbors)
        self.knn.fit(embedd)

    def predict(self, batch_size  = 4096):
        predictions = []
        SPLITS = (len(self.embedd) // batch_size) if (len(self.embedd) % batch_size == 0) else (len(self.embedd) // batch_size + 1)

        for i in range(SPLITS):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(self.embedd))
            batch_data = self.embedd[start: end]

            distance, indices = self.knn.kneighbors()
            for j in range(start, end):
                ids = np.where(distance[j] < 0.6)[0]
                if len(ids) == 0: continue
                split = indices[j, ids]
                if np.any(ids >= len(self.embedd)): continue

                pred = self.df.iloc[split]['posting_id'].values
                predictions.append(pred)
        return predictions




