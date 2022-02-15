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
from torch import nn
from torch import optim
from torch.nn import functional as F
# from torchviz import make_dot
from tqdm import tqdm
from tracking_decorator import TrackingDecorator

from lstm_classifier import LstmClassifier

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

def get_confusion_matrix_dataframe(classifier, data_loader):
    targets = []
    predictions = []
    confusion_matrix = np.zeros((num_classes, num_classes))

    for batch in data_loader:
        input, target = [t.to(device) for t in batch]
        output = classifier(input)
        prediction = F.log_softmax(output, dim=1).argmax(dim=1)

        targets.extend(target.tolist())
        predictions.extend(prediction.tolist())

        for t, p in zip(target.view(-1), prediction.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    # Build confusion matrix, limit to classes actually used
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix, index=LabelEncoder().classes,
                                              columns=LabelEncoder().classes).astype("int64")
    used_columns = (confusion_matrix_dataframe != 0).any(axis=0).where(lambda x: x == True).dropna().keys().tolist()
    used_rows = (confusion_matrix_dataframe != 0).any(axis=1).where(lambda x: x == True).dropna().keys().tolist()
    used_classes = list(dict.fromkeys(used_columns + used_rows))
    confusion_matrix_dataframe = confusion_matrix_dataframe.filter(items=used_classes, axis=0).filter(
        items=used_classes, axis=1)

    return confusion_matrix_dataframe, targets, predictions


def get_metrics(classifier, data_loader):
    confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(classifier, data_loader)

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

class LstmBaseModel:

    def __init__(self, logger, log_path_modelling, log_path_evaluation, train_dataframes, test_dataframes,
                 epochs, learning_rate, patience, dropout, slice_width,
                 lstm_hidden_dimension, lstm_layer_dimension):
        self.logger = logger
        self.log_path_modelling = log_path_modelling
        self.log_path_evaluation = log_path_evaluation

        self.train_dataframes = train_dataframes
        self.test_dataframes = test_dataframes

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.dropout = dropout
        self.slice_width = slice_width

        self.lstm_hidden_dimension = lstm_hidden_dimension
        self.lstm_layer_dimension = lstm_layer_dimension

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
        split_index = 0
        split_labels = []

        overall_validation_accuracy_history = []
        overall_validation_precision_history = []
        overall_validation_recall_history = []
        overall_validation_f1_score_history = []
        overall_validation_cohen_kappa_score_history = []
        overall_validation_matthews_correlation_coefficient_history = []
        overall_epochs = []

        ids = sorted(list(self.train_dataframes.keys()))

        for train_ids, validation_ids in kf.split(ids):
            # Increment split index
            split_index += 1
            split_labels.append("Split " + str(split_index))

            # Validate split
            validation_accuracy_history, validation_precision_history, validation_recall_history, \
            validation_f1_score_history, validation_cohen_kappa_score_history, \
            validation_matthews_correlation_coefficient_history, epoch = self.validate_split(
                split_index=split_index,
                k_folds=k_folds,
                dataframes=self.train_dataframes,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
                slice_width=self.slice_width,
                dropout=self.dropout,
                lstm_hidden_dimension=self.lstm_hidden_dimension,
                lstm_layer_dimension=self.lstm_layer_dimension,
                train_ids=train_ids,
                validation_ids=validation_ids,
                quiet=quiet,
                dry_run=dry_run
            )

            # Aggregate split results
            overall_validation_accuracy_history.append(validation_accuracy_history)
            overall_validation_precision_history.append(validation_precision_history)
            overall_validation_recall_history.append(validation_recall_history)
            overall_validation_f1_score_history.append(validation_f1_score_history)
            overall_validation_cohen_kappa_score_history.append(validation_cohen_kappa_score_history)
            overall_validation_matthews_correlation_coefficient_history.append(
                validation_matthews_correlation_coefficient_history)
            overall_epochs.append(epoch)

        self.model_plotter.plot_split_results(
            logger=self.logger,
            log_path=self.log_path_modelling,
            split_labels=split_labels,
            overall_validation_accuracy_history=overall_validation_accuracy_history,
            overall_validation_precision_history=overall_validation_precision_history,
            overall_validation_recall_history=overall_validation_recall_history,
            overall_validation_f1_score_history=overall_validation_f1_score_history,
            overall_validation_cohen_kappa_score_history=overall_validation_cohen_kappa_score_history,
            overall_validation_matthews_correlation_coefficient_history=overall_validation_matthews_correlation_coefficient_history,
            quiet=quiet
        )

        self.model_logger.log_split_results(
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

        return int(np.mean(overall_epochs))

    @TrackingDecorator.track_time
    def validate_split(self, split_index, k_folds, train_ids, validation_ids, dataframes, epochs,
                       learning_rate, patience, slice_width, dropout, lstm_hidden_dimension, lstm_layer_dimension,
                       quiet, dry_run):
        """
        Validates a single split
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(os.path.join(self.log_path_modelling, "split-" + str(split_index), "models"), exist_ok=True)

        self.logger.log_line("\n Split # " + str(split_index) + "/" + str(k_folds))

        train_dataframes = {id: list(dataframes.values())[id] for id in train_ids}
        validation_dataframes = {id: list(dataframes.values())[id] for id in validation_ids}

        # Create data loader for train
        train_array = self.model_preparator.create_array(train_dataframes)
        train_dataset = self.model_preparator.create_dataset(train_array)
        train_data_loader = self.model_preparator.create_loader(train_dataset, shuffle=False)

        # Create data loader for validation
        validation_array = self.model_preparator.create_array(validation_dataframes)
        validation_dataset = self.model_preparator.create_dataset(validation_array)
        validation_data_loader = self.model_preparator.create_loader(validation_dataset, shuffle=False)

        # Plot target variable distribution
        self.model_plotter.plot_split_distribution(
            logger=self.logger,
            log_path=os.path.join(self.log_path_modelling, "split-" + str(split_index), "plots"),
            train_dataframes=train_dataframes,
            validation_dataframes=validation_dataframes,
            split_index=split_index,
            slice_width=slice_width,
            quiet=quiet
        )

        # Define classifier
        classifier = LstmClassifier(device=device, input_size=slice_width, hidden_dimension=lstm_hidden_dimension,
                                    layer_dimension=lstm_layer_dimension, num_classes=num_classes,
                                    dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        validation_accuracy_max = 0
        validation_precision_max = 0
        validation_recall_max = 0
        validation_f1_score_max = 0
        validation_cohen_kappa_score_max = 0
        validation_matthews_correlation_coefficient_max = 0
        trials = 0

        train_loss_history = []
        train_accuracy_history = []
        train_precision_history = []
        train_recall_history = []
        train_f1_score_history = []
        train_cohen_kappa_score_history = []
        train_matthews_correlation_coefficient_history = []

        validation_loss_history = []
        validation_accuracy_history = []
        validation_precision_history = []
        validation_recall_history = []
        validation_f1_score_history = []
        validation_cohen_kappa_score_history = []
        validation_matthews_correlation_coefficient_history = []

        # Run training loop
        progress_bar = tqdm(iterable=range(1, epochs + 1), unit="epoch", desc="Train model")
        for epoch in progress_bar:

            # Train model
            classifier.train()
            train_epoch_loss = 0
            for i, batch in enumerate(train_data_loader):
                input, target = [t.to(device) for t in batch]
                optimizer.zero_grad()
                output = classifier(input)
                loss = criterion(output, target)
                train_epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # if not os.path.exists(os.path.join(self.log_path_modelling, "model.png")):
                #
                #     # Plot classifier
                #     make_dot(var=output, params=dict(list(classifier.named_parameters()))) \
                #         .render(os.path.join(self.log_path_modelling, "model"), format="png")

            train_epoch_loss /= train_array.shape[0]
            train_loss_history.append(train_epoch_loss)
            classifier.eval()

            with torch.no_grad():
                validation_epoch_loss = 0
                for i, batch in enumerate(validation_data_loader):
                    input, target = [t.to(device) for t in batch]
                    optimizer.zero_grad()
                    output = classifier(input)
                    loss = criterion(output, target)
                    validation_epoch_loss += loss.item()

                validation_epoch_loss /= validation_array.shape[0]
                validation_loss_history.append(validation_epoch_loss)

            # Get metrics for train data
            train_accuracy, \
            train_precision, \
            train_recall, \
            train_f1_score, \
            train_cohen_kappa_score, \
            train_matthews_correlation_coefficient = get_metrics(classifier, train_data_loader)

            train_accuracy_history.append(train_accuracy)
            train_precision_history.append(train_precision)
            train_recall_history.append(train_recall)
            train_f1_score_history.append(train_f1_score)
            train_cohen_kappa_score_history.append(train_cohen_kappa_score)
            train_matthews_correlation_coefficient_history.append(train_matthews_correlation_coefficient)

            # Get metrics for validation data
            validation_accuracy, \
            validation_precision, \
            validation_recall, \
            validation_f1_score, \
            validation_cohen_kappa_score, \
            validation_matthews_correlation_coefficient = get_metrics(classifier, validation_data_loader)

            validation_accuracy_history.append(validation_accuracy)
            validation_precision_history.append(validation_precision)
            validation_recall_history.append(validation_recall)
            validation_f1_score_history.append(validation_f1_score)
            validation_cohen_kappa_score_history.append(validation_cohen_kappa_score)
            validation_matthews_correlation_coefficient_history.append(validation_matthews_correlation_coefficient)

            if not quiet:
                self.logger.log_line("Split " + str(split_index) + " " +
                                     "epoch " + str(epoch) + " " +
                                     "loss " + str(round(train_epoch_loss, 4)).ljust(4, '0') + ", " +
                                     "accuracy " + str(round(validation_accuracy, 2)) + ", " +
                                     "precision " + str(round(validation_precision, 2)) + ", " +
                                     "recall " + str(round(validation_recall, 2)) + ", " +
                                     "f1 score " + str(round(validation_f1_score, 2)) + ", " +
                                     "cohen kappa score " + str(round(validation_cohen_kappa_score, 2)) + ", " +
                                     "matthew correlation coefficient " + str(
                    round(validation_matthews_correlation_coefficient, 2)),
                                     console=False, file=True)

            # Check if Matthews correlation coefficient score increased
            if validation_matthews_correlation_coefficient > validation_matthews_correlation_coefficient_max:
                trials = 0
                validation_accuracy_max = validation_accuracy
                validation_precision_max = validation_precision
                validation_recall_max = validation_recall
                validation_f1_score_max = validation_f1_score
                validation_cohen_kappa_score_max = validation_cohen_kappa_score
                validation_matthews_correlation_coefficient_max = validation_matthews_correlation_coefficient
                torch.save(classifier.state_dict(),
                           os.path.join(self.log_path_modelling, "split-" + str(split_index), "models", "model.pickle"))
            else:
                trials += 1
                if trials >= patience and not quiet:
                    self.logger.log_line("\nNo further improvement after epoch " + str(epoch))
                    break

            if epoch >= epochs:
                self.logger.log_line("\nLast epoch reached")
                break

        progress_bar.close()

        self.model_plotter.plot_training_results(
            logger=self.logger,
            log_path=self.log_path_modelling,
            train_loss_history=train_loss_history,
            validation_loss_history=validation_loss_history,
            train_accuracy_history=train_accuracy_history,
            validation_accuracy_history=validation_accuracy_history,
            validation_precision_history=validation_precision_history,
            validation_recall_history=validation_recall_history,
            validation_f1_score_history=validation_f1_score_history,
            validation_cohen_kappa_score_history=validation_cohen_kappa_score_history,
            validation_matthews_correlation_coefficient_history=validation_matthews_correlation_coefficient_history,
            split_index=split_index,
            quiet=quiet
        )

        self.logger.log_split(
            time_elapsed="{}".format(datetime.now() - start_time),
            k_split=split_index,
            k_folds=k_folds,
            epochs=epoch,
            accuracy=round(validation_accuracy_max, 2),
            precision=round(validation_precision_max, 2),
            recall=round(validation_recall_max, 2),
            f1_score=round(validation_f1_score_max, 2),
            cohen_kappa_score=round(validation_cohen_kappa_score_max, 2),
            matthews_correlation_coefficient=round(validation_matthews_correlation_coefficient_max, 2),
            telegram=not quiet and not dry_run
        )

        return validation_accuracy_history, validation_precision_history, validation_recall_history, \
               validation_f1_score_history, validation_cohen_kappa_score_history, \
               validation_matthews_correlation_coefficient_history, epoch

    @TrackingDecorator.track_time
    def finalize(self, epochs, quiet=False, dry_run=False):
        """
        Trains a final model by using all train dataframes
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(self.log_path_modelling, exist_ok=True)

        # Create data loader for train
        train_array = self.model_preparator.create_array(self.train_dataframes)
        train_dataset = self.model_preparator.create_dataset(train_array)
        train_data_loader = self.model_preparator.create_loader(train_dataset, shuffle=False)

        # Define classifier
        classifier = LstmClassifier(device=device, input_size=self.slice_width,
                                    hidden_dimension=self.lstm_hidden_dimension,
                                    layer_dimension=self.lstm_layer_dimension, num_classes=num_classes,
                                    dropout=self.dropout).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(classifier.parameters(), lr=self.learning_rate)

        # Run training loop
        progress_bar = tqdm(iterable=range(1, epochs + 1), unit="epoch", desc="Train model")
        for epoch in progress_bar:

            # Train model
            classifier.train()
            train_epoch_loss = 0
            for i, batch in enumerate(train_data_loader):
                input, target = [t.to(device) for t in batch]
                optimizer.zero_grad()
                output = classifier(input)
                loss = criterion(output, target)
                train_epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            classifier.eval()

            if not quiet:
                self.logger.log_line("Epoch " + str(epoch) + " loss " + str(round(train_epoch_loss, 4)).ljust(4, '0'),
                                     console=False, file=True)

        progress_bar.close()

        torch.save(classifier.state_dict(), os.path.join(self.log_path_modelling, "model.pickle"))

        self.logger.log_finalization(
            time_elapsed="{}".format(datetime.now() - start_time),
            epochs=epochs,
            telegram=not quiet and not dry_run
        )

    @TrackingDecorator.track_time
    def evaluate(self, clean=False, quiet=False, dry_run=False):
        """
        Evaluates finalized model against test dataframes
        """

        start_time = datetime.now()

        # Make results path
        os.makedirs(self.log_path_evaluation, exist_ok=True)

        # Create data loader
        test_array = self.model_preparator.create_array(self.test_dataframes)
        test_dataset = self.model_preparator.create_dataset(test_array)
        test_data_loader = self.model_preparator.create_loader(test_dataset, shuffle=False)

        # Determine number of linear channels based on slice width
        linear_channels = self.model_preparator.get_linear_channels(self.slice_width)

        # Define classifier
        classifier = LstmClassifier(device=device, input_size=self.slice_width,
                                    hidden_dimension=self.lstm_hidden_dimension,
                                    layer_dimension=self.lstm_layer_dimension, num_classes=num_classes).to(device)
        classifier.load_state_dict(torch.load(os.path.join(self.log_path_modelling, "model.pickle")))
        classifier.eval()

        # Evaluate with test dataloader
        test_accuracy, \
        test_precision, \
        test_recall, \
        test_f1_score, \
        test_cohen_kappa_score, \
        test_matthews_correlation_coefficient = get_metrics(classifier, test_data_loader)

        # Plot confusion matrix
        test_confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(classifier,
                                                                                               test_data_loader)
        ConfusionMatrixPlotter().run(self.logger, os.path.join(self.log_path_evaluation, "plots"),
                                     test_confusion_matrix_dataframe, clean=clean)

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

        return test_accuracy, test_precision, test_recall, test_f1_score, test_cohen_kappa_score, test_matthews_correlation_coefficient
