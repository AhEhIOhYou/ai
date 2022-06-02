import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


class KMeansPredict:
    def __init__(self):
        self.clusters = None

    def get_dataset(self):
        path = "dataset/"
        housing = pd.read_csv(path + "housing.csv", index_col=0)
        return housing

    def fetch_housing_data(self, data):
        data.columns = data.columns.astype(int)
        data_size = data.shape[1]
        train_size = int(np.round(data_size * (30 / 100)))

        ready_data = {
            "X_train": np.empty(shape=[train_size, 5], dtype=int),
            "y_train": np.empty(shape=train_size, dtype=int),
            "X_test": np.empty(shape=[data_size - train_size, 5], dtype=int),
            "y_test": np.empty(shape=data_size - train_size, dtype=int),
            "names": np.array([
                "income",
                "rooms_per_house",
                "near_ocean",
                "near_bay",
                "1h_ocean",
                "new_house",
                "people_per_house",
                "latitude",
                "longitude",
            ]),
            "clusters": np.array([
                "<50000$",
                "50 000-150 000 $",
                "150 000-250 000 $",
                "250 000-350 000 $",
                "350 000-450 000 $",
                "450 000-550 000 $"
            ])
        }

        for i in range(train_size):
            ready_data["y_train"][i] = data[i]["median_house_value"]
            ready_data["X_train"][i] = np.ceil(data[i].drop("median_house_value") * 10)

        for i in range(data_size - train_size):
            ready_data["y_test"][i] = data[i]["median_house_value"]
            ready_data["X_test"][i] = np.ceil(data[i].drop("median_house_value") * 10)

        return ready_data

    def predict(self, data):
        kmeans = KMeans(n_clusters=6)
        clusters = kmeans.fit_predict(data["X_train"])
        self.clusters = clusters

    def print_matrix(self, data):
        labels = np.zeros_like(self.clusters)

        for i in range(6):
            mask = (self.clusters == i)
            labels[mask] = mode(data["y_train"][mask])[0]

        conf_matrix = confusion_matrix(data["y_train"], labels)
        sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=data["clusters"], yticklabels=data["clusters"])
        plt.show()


item = KMeansPredict()
dataset = item.get_dataset()
data = item.fetch_housing_data(dataset)
item.predict(data)
item.print_matrix(data)
