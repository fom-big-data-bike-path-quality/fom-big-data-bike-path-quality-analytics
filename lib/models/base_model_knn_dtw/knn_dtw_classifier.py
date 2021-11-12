import sys

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import mode
from tqdm import tqdm


def build_distance_matrix(train_data, test_data, max_warping_window, subsample_step):
    # Compute the distance matrix
    distance_matrix_count = 0

    # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
    # when x and y are the same array
    if (np.array_equal(train_data, test_data)):
        train_data_shape = np.shape(train_data)
        distance_matrix = np.zeros((train_data_shape[0] * (train_data_shape[0] - 1)) // 2, dtype=np.double)

        with tqdm(total=train_data_shape[0] * (train_data_shape[0] - 1), desc="Build distance matrix") as progress:
            for i in range(0, train_data_shape[0] - 1):
                for j in range(i + 1, train_data_shape[0]):
                    distance_matrix[distance_matrix_count] = dtw_distance(
                        train_data[i, ::subsample_step],
                        test_data[j, ::subsample_step],
                        max_warping_window=max_warping_window
                    )

                    distance_matrix_count += 1
                    progress.update(1)

        # Convert to squareform
        distance_matrix = squareform(distance_matrix)
        return distance_matrix

    # Compute full distance matrix of dtw distances between x and y
    else:
        train_data_shape = np.shape(train_data)
        test_data_shape = np.shape(test_data)
        distance_matrix = np.zeros((test_data_shape[0], train_data_shape[0]))

        with tqdm(total=train_data_shape[0] * test_data_shape[0]) as progress:
            for i in range(0, test_data_shape[0]):
                for j in range(0, train_data_shape[0]):
                    distance_matrix[i, j] = dtw_distance(
                        test_data[i, ::subsample_step],
                        train_data[j, ::subsample_step],
                        max_warping_window=max_warping_window
                    )

                    progress.update(1)

        return distance_matrix


def dtw_distance(timeseries_array_a, timeseries_array_b, max_warping_window, d=lambda x, y: abs(x - y)):
    # Create cost matrix via broadcasting with large int
    M, N = len(timeseries_array_a), len(timeseries_array_b)
    cost = sys.maxsize * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(timeseries_array_a[0], timeseries_array_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(timeseries_array_a[i], timeseries_array_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(timeseries_array_a[0], timeseries_array_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(timeseries_array_a[i], timeseries_array_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


class KnnDtwClassifier:

    def __init__(self, k, max_warping_window, subsample_step=1):
        self.k = k
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, input_data):
        distance_matrix = build_distance_matrix(
            train_data=input_data,
            input_data=self.train_data,
            max_warping_window=self.max_warping_window,
            subsample_step=self.subsample_step
        )

        knn_indices = distance_matrix.argsort()[:, :self.k]
        knn_labels = self.train_labels[knn_indices]

        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.k

        return mode_label.astype(int).ravel(), mode_proba.ravel()
