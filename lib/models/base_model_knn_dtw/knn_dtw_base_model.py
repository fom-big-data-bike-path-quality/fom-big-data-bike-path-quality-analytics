import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from label_encoder import LabelEncoder
from model_evaluator import ModelEvaluator
from model_logger import ModelLogger
from model_plotter import ModelPlotter
from model_preparator import ModelPreparator
from sklearn.model_selection import KFold
from tracking_decorator import TrackingDecorator

from knn_dtw_classifier import KnnDtwClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Control sources of randomness
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Number of classes
num_classes = LabelEncoder().num_classes()


#
# Evaluation
#

def get_confusion_matrix_dataframe(classifier, input, target):
    targets = []
    predictions = []
    confusion_matrix = np.zeros((num_classes, num_classes))

    prediction, probability = classifier.predict(input)

    targets.extend(target.astype(int).tolist())
    predictions.extend(prediction.astype(int).tolist())

    for t, p in zip(target.astype(int), prediction.astype(int)):
        confusion_matrix[t, p] += 1

    # Build confusion matrix, limit to classes actually used
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix, index=LabelEncoder().classes,
                                              columns=LabelEncoder().classes).astype("int64")
    used_columns = (confusion_matrix_dataframe != 0).any(axis=0).where(lambda x: x == True).dropna().keys().tolist()
    used_rows = (confusion_matrix_dataframe != 0).any(axis=1).where(lambda x: x == True).dropna().keys().tolist()
    used_classes = list(dict.fromkeys(used_columns + used_rows))
    confusion_matrix_dataframe = confusion_matrix_dataframe.filter(items=used_classes, axis=0).filter(
        items=used_classes, axis=1)

    return confusion_matrix_dataframe, targets, predictions


def get_metrics(classifier, data, labels):
    confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(
        classifier=classifier,
        input=data,
        target=labels
    )

    model_evaluator = ModelEvaluator()

    accuracy = model_evaluator.get_accuracy(confusion_matrix_dataframe)
    precision = model_evaluator.get_precision(confusion_matrix_dataframe)
    recall = model_evaluator.get_recall(confusion_matrix_dataframe)
    f1_score = model_evaluator.get_f1_score(confusion_matrix_dataframe)
    cohen_kappa_score = model_evaluator.get_cohen_kappa_score(targets, predictions)
    matthew_correlation_coefficient = model_evaluator.get_matthews_corrcoef_score(targets, predictions)

    return accuracy, precision, recall, f1_score, cohen_kappa_score, matthew_correlation_coefficient


#
# Main
#

class KnnDtwBaseModel:

    def __init__(self, logger, log_path_modelling, log_path_evaluation, train_dataframes, test_dataframes, slice_width):
        self.logger = logger
        self.log_path_modelling = log_path_modelling
        self.log_path_evaluation = log_path_evaluation

        self.train_dataframes = train_dataframes
        self.test_dataframes = test_dataframes

        self.slice_width = slice_width

        self.model_logger = ModelLogger()
        self.model_plotter = ModelPlotter()
        self.model_preparator = ModelPreparator()
        self.model_evaluator = ModelEvaluator()

    @TrackingDecorator.track_time
    def validate(self, k_folds, random_state=0, quiet=False, dry_run=False):
        """
        Validates the model by folding all train dataframes
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(self.log_path_modelling, exist_ok=True)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        fold_index = 0
        fold_labels = []

        overall_validation_accuracy = []
        overall_validation_precision = []
        overall_validation_recall = []
        overall_validation_f1_score = []
        overall_validation_cohen_kappa_score = []
        overall_validation_matthew_correlation_coefficient = []

        ids = sorted(list(self.train_dataframes.keys()))

        for train_ids, validation_ids in kf.split(ids):
            # Increment fold index
            fold_index += 1
            fold_labels.append("Fold " + str(fold_index))

            # Validate fold
            validation_accuracy, validation_precision, validation_recall, validation_f1_score, \
            validation_cohen_kappa_score, validation_matthew_correlation_coefficient = self.validate_fold(
                fold_index=fold_index,
                k_folds=k_folds,
                dataframes=self.train_dataframes,
                slice_width=self.slice_width,
                train_ids=train_ids,
                validation_ids=validation_ids,
                quiet=quiet,
                dry_run=dry_run
            )

            # Aggregate fold results
            overall_validation_accuracy.append(validation_accuracy)
            overall_validation_precision.append(validation_precision)
            overall_validation_recall.append(validation_recall)
            overall_validation_f1_score.append(validation_f1_score)
            overall_validation_cohen_kappa_score.append(validation_cohen_kappa_score)
            overall_validation_matthew_correlation_coefficient.append(
                validation_matthew_correlation_coefficient)

        # TODO Plot

        # TODO Log

        self.logger.log_validation(
            time_elapsed="{}".format(datetime.now() - start_time),
            log_path_modelling=self.log_path_modelling,
            telegram=not quiet and not dry_run
        )

        return 0

    @TrackingDecorator.track_time
    def validate_fold(self, fold_index, k_folds, train_ids, validation_ids, dataframes, slice_width, quiet, dry_run):
        """
        Validates a single fold
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(os.path.join(self.log_path_modelling, "models", "fold-" + str(fold_index)), exist_ok=True)

        self.logger.log_line("\n Fold # " + str(fold_index) + "\n")

        train_dataframes = {id: list(dataframes.values())[id] for id in train_ids}
        validation_dataframes = {id: list(dataframes.values())[id] for id in validation_ids}

        # Split data and labels for train
        train_array = self.model_preparator.create_array(train_dataframes)
        train_data, train_labels = self.model_preparator.split_data_and_labels(train_array)

        # Split data and labels for validation
        validation_array = self.model_preparator.create_array(validation_dataframes)
        validation_data, validation_labels = self.model_preparator.split_data_and_labels(validation_array)

        # Plot target variable distribution
        self.model_plotter.plot_fold_distribution(
            logger=self.logger,
            log_path=self.log_path_modelling,
            train_dataframes=train_dataframes,
            validation_dataframes=validation_dataframes,
            fold_index=fold_index,
            slice_width=slice_width,
            quiet=quiet
        )

        # Define classifier
        classifier = KnnDtwClassifier(k=5, max_warping_window=10)
        classifier.fit(train_data, train_labels)

        # Get metrics for train data
        train_accuracy, \
        train_precision, \
        train_recall, \
        train_f1_score, \
        train_cohen_kappa_score, \
        train_matthew_correlation_coefficient = get_metrics(classifier, train_data, train_labels)

        # Get metrics for validation data
        validation_accuracy, \
        validation_precision, \
        validation_recall, \
        validation_f1_score, \
        validation_cohen_kappa_score, \
        validation_matthew_correlation_coefficient = get_metrics(classifier, validation_data, validation_labels)

        self.logger.log_fold(
            time_elapsed="{}".format(datetime.now() - start_time),
            k_fold=fold_index,
            k_folds=k_folds,
            epochs=None,
            accuracy=round(validation_accuracy, 2),
            precision=round(validation_precision, 2),
            recall=round(validation_recall, 2),
            f1_score=round(validation_f1_score, 2),
            cohen_kappa_score=round(validation_cohen_kappa_score, 2),
            matthew_correlation_coefficient=round(validation_matthew_correlation_coefficient, 2),
            telegram=not quiet and not dry_run
        )

        return validation_accuracy, validation_precision, validation_recall, validation_f1_score, \
               validation_cohen_kappa_score, validation_matthew_correlation_coefficient

    @TrackingDecorator.track_time
    def finalize(self, epochs, quiet=False, dry_run=False):
        """
        Trains a final model by using all train dataframes
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(self.log_path_modelling, exist_ok=True)

        # Split data and labels for train
        train_array = self.model_preparator.create_array(self.train_dataframes)
        train_data, train_labels = self.model_preparator.split_data_and_labels(train_array)

        # Define classifier
        classifier = KnnDtwClassifier(k=5, max_warping_window=10)
        classifier.fit(train_data, train_labels)

        self.logger.log_finalization(
            time_elapsed="{}".format(datetime.now() - start_time),
            epochs=None,
            telegram=not quiet and not dry_run
        )

    @TrackingDecorator.track_time
    def evaluate(self, clean=False, quiet=False, dry_run=False):
        pass