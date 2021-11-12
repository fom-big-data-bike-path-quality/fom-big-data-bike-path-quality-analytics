import numpy as np
from dtaidistance import dtw
from scipy.stats import mode
from tqdm import tqdm


def build_distance_matrix(train_data, test_data, subsample_step, max_warping_window, use_pruning):
    train_data_shape = np.shape(train_data)
    test_data_shape = np.shape(test_data)
    distance_matrix = np.zeros((test_data_shape[0], train_data_shape[0]))

    with tqdm(total=train_data_shape[0] * test_data_shape[0]) as progress:
        for i in range(0, test_data_shape[0]):
            for j in range(0, train_data_shape[0]):
                distance_matrix[i, j] = dtw_distance(
                    test_data[i, ::subsample_step],
                    train_data[j, ::subsample_step],
                    max_warping_window=max_warping_window,
                    use_pruning=use_pruning
                )

                progress.update(1)

    return distance_matrix


def dtw_distance(timeseries_array_a, timeseries_array_b, max_warping_window, use_pruning=False):
    return dtw.distance(timeseries_array_a, timeseries_array_b, window=max_warping_window, use_pruning=use_pruning)


class KnnDtwClassifier:

    def __init__(self, k, subsample_step=1, max_warping_window=10, use_pruning=False):
        self.k = k
        self.subsample_step = subsample_step
        self.max_warping_window = max_warping_window
        self.use_pruning = use_pruning

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, input_data):
        self.distance_matrix = build_distance_matrix(
            train_data=self.train_data,
            test_data=input_data,
            subsample_step=self.subsample_step,
            max_warping_window=self.max_warping_window,
            use_pruning=self.use_pruning
        )

        knn_indices = self.distance_matrix.argsort()[:, :self.k]
        knn_labels = self.train_labels[knn_indices]

        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_probability = mode_data[1] / self.k

        return mode_label.astype(int).ravel(), mode_probability.ravel()
