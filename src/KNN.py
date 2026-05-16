__authors__ = ["1748951", "1755033", "1703660"]
__group__ = "11"

import numpy as np
from scipy.spatial.distance import cdist

from typing import Callable

DistanceMetric = Callable[[np.ndarray, np.ndarray], np.ndarray]

EUCLIDEAN_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "euclidean")
MANHATTAN_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "cityblock")
COSINE_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "cosine")


class KNN:
    def __init__(
        self, train_data: np.ndarray, labels: np.ndarray, distance_fn: DistanceMetric
    ):
        self._init_train(train_data)
        self.distance_fn = distance_fn
        self.labels = np.array(labels)

    def _init_train(self, train_data: np.ndarray):
        train_data = np.array(train_data, dtype=float)
        self.original_shape = train_data.shape[1:3]  # (m, n)
        self.train_data = self._extract_features(train_data)

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=float)
        return data.reshape(data.shape[0], -1)

    def get_k_neighbours(self, test_data: np.ndarray, k: int) -> np.ndarray:
        test_data = np.array(test_data, dtype=float)
        test_feats = self._extract_features(test_data)

        distances = self.distance_fn(test_feats, self.train_data)

        sorted_idxs = np.argsort(distances, axis=-1)
        self.neighbors = self.labels[sorted_idxs[:, :k]]
        return self.neighbors

    def get_class(self) -> np.ndarray:
        # predictions = []
        #
        # for row in self.neighbors:
        #     vals, count = np.unique(row, return_counts=True)
        #     best_val = vals[np.argmax(count)]
        #     predictions.append(best_val)
        #
        # return np.array(predictions)
        predictions = []

        for row in self.neighbors:
            row_list = list(row)
            best_val = None
            max_votes = -1

            already_counted = []
            for label in row_list:
                if label not in already_counted:
                    votes = row_list.count(label)
                    if votes > max_votes:
                        max_votes = votes
                        best_val = label
                    already_counted.append(label)

            predictions.append(best_val)

        return np.array(predictions)

    def predict(self, test_data: np.ndarray, k: int) -> np.ndarray:
        self.get_k_neighbours(test_data, k)
        return self.get_class()

    # def _extract_features(self, data):
    #     """
    #     extracts features from a set of images according to options['feature_mode'].
    #     Arguments:
    #         data (np.ndarray): shape (p, m, n, 3)
    #     Returns:
    #         np.ndarray: shape (p, d) where D depends on the chosen mode
    #     Raises:
    #         ValueError if no valid 'feature_mode' is set in options
    #     """
    #
    #     mode = self.options.feature_mode
    #     n = data.shape[0]
    #
    #     if mode == "pixels":
    #         # the oririginal behaviour we had: flatten all RGB pixels
    #         return data.reshape(n, -1)
    #
    #     elif mode == "reduced":
    #         # we downsample each image to a smaller resolution
    #         scale = self.options.reduction_scale
    #         M, N = data.shape[1], data.shape[2]
    #         new_M = max(1, int(M * scale))
    #         new_N = max(1, int(N * scale))
    #
    #         # simple block-average downsampling (this way no external libraries are needed)
    #         step_M = max(1, M // new_M)
    #         step_N = max(1, N // new_N)
    #         reduced = data[:, ::step_M, ::step_N, :]
    #         return reduced.reshape(n, -1)
    #
    #     elif mode == "stats":
    #         # 4 statistics x 3 channels = 12 features per image
    #         features = np.zeros((n, 12))
    #         for c in range(3):
    #             channel = data[:, :, :, c].reshape(n, -1)
    #             features[:, c * 4 + 0] = channel.mean(axis=1)
    #             features[:, c * 4 + 1] = channel.std(axis=1)
    #             features[:, c * 4 + 2] = channel.min(axis=1)
    #             features[:, c * 4 + 3] = channel.max(axis=1)
    #         return features
    #
    #     else:
    #         raise ValueError(
    #             f"feature_mode '{mode}' not recognised. Use 'pixels', 'reduced' or 'stats'."
    #         )

    # def get_k_neighbours(self, test_data, k):
    #     """
    #     calculates the k nearest neighbours for each point in test_data.
    #     Supports distance metrics via options['distance']:
    #         'euclidean' : L2 distance (default)
    #         'manhattan' : L1 distance (upgrade)
    #         'cosine'    : cosine dissimilarity (upgrade)
    #     """
    #     test_data = np.array(test_data, dtype=float)
    #     test_feats = self._extract_features(test_data)
    #
    #     metric = self.options.distance
    #
    #     if metric == "euclidean":
    #         distances = cdist(test_feats, self.train_data, "euclidean")
    #     elif metric == "manhattan":
    #         distances = cdist(test_feats, self.train_data, "cityblock")
    #     elif metric == "cosine":
    #         distances = cdist(test_feats, self.train_data, "cosine")
    #     else:
    #         raise ValueError(
    #             f"Distance '{metric}' not recognised. Use 'euclidean', 'manhattan' or 'cosine'."
    #         )
    #
    #     sorted_idxs = np.argsort(distances, axis=1)
    #     self.neighbors = self.labels[sorted_idxs[:, :k]]

    # def get_class(self):
    #     """
    #     Get the class by maximum voting
    #     :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
    #             (i.e. the class at which that row belongs)
    #     """
    #     #######################################################
    #     ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    #     ##  AND CHANGE FOR YOUR OWN CODE
    #     #######################################################
    #     predictions = []
    #
    #     for row in self.neighbors:
    #         row_list = list(row)
    #         best_val = None
    #         max_votes = -1
    #
    #         already_counted = []
    #         for label in row_list:
    #             if label not in already_counted:
    #                 votes = row_list.count(label)
    #                 if votes > max_votes:
    #                     max_votes = votes
    #                     best_val = label
    #                 already_counted.append(label)
    #
    #         predictions.append(best_val)
    #
    #     return np.array(predictions)

    # def predict(self, test_data, k):
    #     """
    #     predicts the class at which each element in test_data belongs to
    #     :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
    #     :param k: the number of neighbors to look at
    #     :return: the output form get_class a Nx1 vector with the predicted shape for each test image
    #     """
    #
    #     self.get_k_neighbours(test_data, k)
    #     return self.get_class()


# """
# EXAMPLE OF USE
#
# knn = KNN(train_data, labels)
# knn.options = {'feature_mode': 'reduced', 'reduction_scale': 0.5, 'distance': 'manhattan'}
# knn.predict(test_data, k=5)
#
# """
