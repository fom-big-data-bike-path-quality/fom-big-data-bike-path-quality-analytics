import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from confusion_matrix_plotter import ConfusionMatrixPlotter
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

def get_confusion_matrix_dataframe(classifier, input, target, k):
    targets = []
    predictions = []
    confusion_matrix = np.zeros((num_classes, num_classes))

    prediction, probability = classifier.predict(input, k)

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


def get_metrics(classifier, data, labels, k):
    confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(
        classifier=classifier,
        input=data,
        target=labels,
        k=k
    )

    model_evaluator = ModelEvaluator()

    accuracy = model_evaluator.get_accuracy(confusion_matrix_dataframe)
    precision = model_evaluator.get_precision(confusion_matrix_dataframe)
    recall = model_evaluator.get_recall(confusion_matrix_dataframe)
    f1_score = model_evaluator.get_f1_score(confusion_matrix_dataframe)
    cohen_kappa_score = model_evaluator.get_cohen_kappa_score(targets, predictions)
    matthews_correlation_coefficient = model_evaluator.get_matthews_corrcoef_score(targets, predictions)

    return accuracy, precision, recall, f1_score, cohen_kappa_score, matthews_correlation_coefficient


#
# Main
#

class KnnDtwBaseModel:

    def __init__(self, logger, log_path_modelling, log_path_evaluation, train_dataframes, test_dataframes,
                 k_nearest_neighbors, subsample_step, max_warping_window, slice_width):
        self.logger = logger
        self.log_path_modelling = log_path_modelling
        self.log_path_evaluation = log_path_evaluation

        self.train_dataframes = train_dataframes
        self.test_dataframes = test_dataframes

        self.k_nearest_neighbors = k_nearest_neighbors
        self.subsample_step = subsample_step
        self.max_warping_window = max_warping_window
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

        overall_validation_accuracy_history = []
        overall_validation_precision_history = []
        overall_validation_recall_history = []
        overall_validation_f1_score_history = []
        overall_validation_cohen_kappa_score_history = []
        overall_validation_matthews_correlation_coefficient_history = []

        ids = sorted(list(self.train_dataframes.keys()))

        for train_ids, validation_ids in kf.split(ids):
            # Increment fold index
            fold_index += 1
            fold_labels.append("Fold " + str(fold_index))

            # Validate fold
            k, validation_accuracy, validation_precision, validation_recall, validation_f1_score, \
            validation_cohen_kappa_score, validation_matthews_correlation_coefficient = self.validate_fold(
                fold_index=fold_index,
                k_folds=k_folds,
                dataframes=self.train_dataframes,
                slice_width=self.slice_width,
                train_ids=train_ids,
                validation_ids=validation_ids,
                quiet=quiet,
                dry_run=dry_run
            )

            self.logger.log_line(f"best validation with k={k} : {validation_matthews_correlation_coefficient}")

            # Aggregate fold results
            overall_validation_accuracy_history.append(validation_accuracy)
            overall_validation_precision_history.append(validation_precision)
            overall_validation_recall_history.append(validation_recall)
            overall_validation_f1_score_history.append(validation_f1_score)
            overall_validation_cohen_kappa_score_history.append(validation_cohen_kappa_score)
            overall_validation_matthews_correlation_coefficient_history.append(
                validation_matthews_correlation_coefficient)

        self.model_plotter.plot_fold_results_hist(
            logger=self.logger,
            log_path=self.log_path_modelling,
            fold_labels=fold_labels,
            overall_validation_accuracy_history=overall_validation_accuracy_history,
            overall_validation_precision_history=overall_validation_precision_history,
            overall_validation_recall_history=overall_validation_recall_history,
            overall_validation_f1_score_history=overall_validation_f1_score_history,
            overall_validation_cohen_kappa_score_history=overall_validation_cohen_kappa_score_history,
            overall_validation_matthews_correlation_coefficient_history=overall_validation_matthews_correlation_coefficient_history,
            quiet=quiet
        )

        self.model_logger.log_fold_results(
            logger=self.logger,
            overall_validation_accuracy_history=overall_validation_accuracy_history,
            overall_validation_precision_history=overall_validation_precision_history,
            overall_validation_recall_history=overall_validation_recall_history,
            overall_validation_f1_score_history=overall_validation_f1_score_history,
            overall_validation_cohen_kappa_score_history=overall_validation_cohen_kappa_score_history,
            overall_validation_matthews_correlation_coefficient_history=overall_validation_matthews_correlation_coefficient_history,
            quiet=quiet)

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
        os.makedirs(os.path.join(self.log_path_modelling, "fold-" + str(fold_index), "models"), exist_ok=True)

        self.logger.log_line("\n Fold # " + str(fold_index) + "/" + str(k_folds))

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
            log_path=os.path.join(self.log_path_modelling, "fold-" + str(fold_index), "plots"),
            train_dataframes=train_dataframes,
            validation_dataframes=validation_dataframes,
            fold_index=fold_index,
            slice_width=slice_width,
            quiet=quiet
        )

        # Define classifier
        classifier = KnnDtwClassifier(k=self.k_nearest_neighbors, subsample_step=1, max_warping_window=10,
                                      use_pruning=True)
        classifier.fit(train_data, train_labels)

        validation_k_list = []
        validation_accuracy_list = []
        validation_precision_list = []
        validation_recall_list = []
        validation_f1_score_list = []
        validation_cohen_kappa_score_list = []
        validation_matthews_correlation_coefficient_list = []

        # Iterate over hyper-parameter configurations
        for k in range(1, self.k_nearest_neighbors + 1):
            # Get metrics for validation data
            validation_accuracy, \
            validation_precision, \
            validation_recall, \
            validation_f1_score, \
            validation_cohen_kappa_score, \
            validation_matthews_correlation_coefficient = get_metrics(
                classifier=classifier,
                data=validation_data,
                labels=validation_labels,
                k=k
            )

            # # Plot distance matrix
            # distance_matrix_dataframe = pd.DataFrame(data=classifier.distance_matrix.astype(int))
            # DistanceMatrixPlotter().run(
            #     logger=self.logger,
            #     results_path=os.path.join(self.log_path_modelling, "fold-" + str(fold_index), "plots"),
            #     distance_matrix_dataframe=distance_matrix_dataframe,
            #     clean=False,
            #     quiet=False
            # )

            np.save(os.path.join(self.log_path_modelling, "fold-" + str(fold_index), "models", "model"),
                    classifier.distance_matrix)

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
                matthews_correlation_coefficient=round(validation_matthews_correlation_coefficient, 2),
                telegram=not quiet and not dry_run
            )

            validation_k_list.append(k)
            validation_accuracy_list.append(validation_accuracy)
            validation_precision_list.append(validation_precision)
            validation_recall_list.append(validation_recall)
            validation_f1_score_list.append(validation_f1_score)
            validation_cohen_kappa_score_list.append(validation_cohen_kappa_score)
            validation_matthews_correlation_coefficient_list.append(validation_matthews_correlation_coefficient)

        # Determine with which k the best result has been created
        index_best = validation_matthews_correlation_coefficient_list.index(
            max(validation_matthews_correlation_coefficient_list))
        k_best = index_best + 1

        return k_best, validation_accuracy_list[index_best], validation_precision_list[index_best], \
               validation_recall_list[index_best], validation_f1_score_list[index_best], \
               validation_cohen_kappa_score_list[index_best], validation_matthews_correlation_coefficient_list[
                   index_best]

    @TrackingDecorator.track_time
    def finalize(self, epochs, quiet=False, dry_run=False):
        """
        Trains a final model by using all train dataframes (not necessary since kNN is instance based)
        """
        pass

    @TrackingDecorator.track_time
    def evaluate(self, clean=False, quiet=False, dry_run=False):
        """
        Evaluates finalized model against test dataframes
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(self.log_path_modelling, exist_ok=True)
        os.makedirs(self.log_path_evaluation, exist_ok=True)

        # Split data and labels for train
        train_array = self.model_preparator.create_array(self.train_dataframes)
        train_data, train_labels = self.model_preparator.split_data_and_labels(train_array)

        # Split data and labels for validation
        test_array = self.model_preparator.create_array(self.test_dataframes)
        test_data, test_labels = self.model_preparator.split_data_and_labels(test_array)

        # Define classifier
        classifier = KnnDtwClassifier(k=self.k_nearest_neighbors, subsample_step=1, max_warping_window=10,
                                      use_pruning=True)
        classifier.fit(train_data, train_labels)

        test_k_list = []
        test_accuracy_list = []
        test_precision_list = []
        test_recall_list = []
        test_f1_score_list = []
        test_cohen_kappa_score_list = []
        test_matthews_correlation_coefficient_list = []

        # Iterate over hyper-parameter configurations
        for k in range(1, self.k_nearest_neighbors):
            # Get metrics for test data
            test_accuracy, \
            test_precision, \
            test_recall, \
            test_f1_score, \
            test_cohen_kappa_score, \
            test_matthews_correlation_coefficient = get_metrics(classifier, test_data, test_labels, k)

            # # Plot distance matrix
            # distance_matrix_dataframe = pd.DataFrame(data=classifier.distance_matrix.astype(int))
            # DistanceMatrixPlotter().run(
            #     logger=self.logger,
            #     results_path=self.log_path_modelling,
            #     distance_matrix_dataframe=distance_matrix_dataframe,
            #     clean=False,
            #     quiet=False
            # )

            np.save(os.path.join(self.log_path_modelling, "model"), classifier.distance_matrix)

            # Plot confusion matrix
            test_confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(
                classifier=classifier,
                input=test_data,
                target=test_labels,
                k=k
            )
            ConfusionMatrixPlotter().run(
                logger=self.logger,
                results_path=os.path.join(self.log_path_evaluation, "plots"),
                confusion_matrix_dataframe=test_confusion_matrix_dataframe,
                file_name="confusion_matrix_k" + str(k),
                clean=clean
            )

            self.logger.log_evaluation(
                time_elapsed="{}".format(datetime.now() - start_time),
                log_path_evaluation=self.log_path_evaluation,
                test_accuracy=test_accuracy,
                test_precision=test_precision,
                test_recall=test_recall,
                test_f1_score=test_f1_score,
                test_cohen_kappa_score=test_cohen_kappa_score,
                test_matthews_correlation_coefficient=test_matthews_correlation_coefficient,
                telegram=not quiet and not dry_run
            )

            test_k_list.append(k)
            test_accuracy_list.append(test_accuracy)
            test_precision_list.append(test_precision)
            test_recall_list.append(test_recall)
            test_f1_score_list.append(test_f1_score)
            test_cohen_kappa_score_list.append(test_cohen_kappa_score)
            test_matthews_correlation_coefficient_list.append(test_matthews_correlation_coefficient)

        return test_k_list, test_accuracy_list, test_precision_list, test_recall_list, test_f1_score_list, \
               test_cohen_kappa_score_list, test_matthews_correlation_coefficient_list
