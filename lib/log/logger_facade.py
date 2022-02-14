import os
import shutil

from console_logger import ConsoleLogger
from file_logger import FileLogger
from telegram_logger import TelegramLogger


def get_split_emoji(split_index):
    emojis = ["🍙", "🧄", "🧅", "🌽", "🫑", "🥦", "🍆", "🌶️", "🧆", "🥙"]
    return emojis[split_index % len(emojis)]


class LoggerFacade:

    def __init__(self, results_path, console=False, file=False, telegram=False):
        self.results_path = results_path
        self.console = console
        self.file = file
        self.telegram = telegram

        shutil.rmtree(results_path, ignore_errors=True)

    def log_line(self, message, images=None, console=None, file=None, telegram=None):
        if console or (console is None and self.console):
            ConsoleLogger().log_line(message)
        if file or (file is None and self.file):
            FileLogger().log_line(self.results_path, message)
        if telegram or (telegram is None and self.telegram):
            TelegramLogger().log_line(self, message, images)

    def log_training_start(self, device_name, training_start_time_string, clean, quiet, transient, dry_run,
                           skip_data_understanding, skip_validation, window_step, down_sampling_factor, model_name,
                           k_folds, k_nearest_neighbors, epochs, learning_rate, patience, slice_width, dropout,
                           lstm_hidden_dimension, lstm_layer_dimension, measurement_speed_limit, test_size,
                           random_state, telegram=None):
        message = "Training started with parameters " + \
                  "\n* device name " + str(device_name) + \
                  "\n* start time " + str(training_start_time_string) + \
                  "\n* clean " + str(clean) + \
                  "\n* quiet " + str(quiet) + \
                  "\n* transient " + str(transient) + \
                  "\n* dry-run " + str(dry_run) + \
                  "\n* skip data understanding " + str(skip_data_understanding) + \
                  "\n* skip validation " + str(skip_validation) + \
                  "\n* window step " + str(window_step) + \
                  "\n* down-sampling factor " + str(down_sampling_factor) + \
                  "\n* model name " + model_name + \
                  "\n* k-folds " + str(k_folds) + \
                  "\n* k-nearest-neighbors " + str(k_nearest_neighbors) + \
                  "\n* epochs " + str(epochs) + \
                  "\n* learning rate " + str(learning_rate) + \
                  "\n* patience " + str(patience) + \
                  "\n* slice width " + str(slice_width) + \
                  "\n* dropout " + str(dropout) + \
                  "\n* lstm-hidden-dimension " + str(lstm_hidden_dimension) + \
                  "\n* lstm-layer-dimension " + str(lstm_layer_dimension) + \
                  "\n* measurement speed limit " + str(measurement_speed_limit) + \
                  "\n* test size " + str(test_size) + \
                  "\n* random state " + str(random_state)

        self.log_line(message=message, telegram=telegram)

    def log_modelling_start(self, model_name, train_dataframes, resampled_train_dataframes, test_dataframes,
                            telegram=None):

        percentage = round(len(resampled_train_dataframes) / len(train_dataframes), 2) * 100

        if len(train_dataframes) == len(resampled_train_dataframes):
            message = "Modelling started with " + model_name + \
                      "\n* train dataframes " + str(len(train_dataframes)) + \
                      "\n* test dataframes " + str(len(test_dataframes))
        else:
            message = "Modelling started with " + model_name + \
                      "\n* train dataframes " + str(len(train_dataframes)) + " down-sampled to " + \
                      str(len(resampled_train_dataframes)) + " (" + str(percentage) + "%)" + \
                      "\n* test dataframes " + str(len(test_dataframes))

        self.log_line(message=message, telegram=telegram)

    def log_split(self, time_elapsed, k_split, k_folds, epochs, accuracy, precision, recall, f1_score,
                  cohen_kappa_score, matthews_correlation_coefficient, telegram=None):

        if epochs is None:
            message = get_split_emoji(k_split) + " Split " + str(k_split) + "/" + str(k_folds) + \
                  " finished in " + time_elapsed + " with validation metrics" + \
                  "\n* accuracy " + str(accuracy) + \
                  "\n* precision " + str(precision) + \
                  "\n* recall " + str(recall) + \
                  "\n* f1 score " + str(f1_score) + \
                  "\n* cohen's kappa score " + str(cohen_kappa_score) + \
                  "\n* matthews correlation coefficient " + str(matthews_correlation_coefficient)
        else:
            message = get_split_emoji(k_split) + " Split " + str(k_split) + "/" + str(k_folds) + \
                      " finished after " + str(epochs) + " epochs in " + time_elapsed + " with validation metrics" + \
                      "\n* accuracy " + str(accuracy) + \
                      "\n* precision " + str(precision) + \
                      "\n* recall " + str(recall) + \
                      "\n* f1 score " + str(f1_score) + \
                      "\n* cohen's kappa score " + str(cohen_kappa_score) + \
                      "\n* matthews correlation coefficient " + str(matthews_correlation_coefficient)

        self.log_line(message=message, telegram=telegram)

    def log_validation(self, time_elapsed, log_path_modelling, telegram=None):

        message = "🍱 Validation finished in " + time_elapsed

        file_path = os.path.join(log_path_modelling, "plots", "overall-f1-score.png")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f1_score_file:
                self.log_line(message=message, images=[f1_score_file], telegram=telegram)

    def log_finalization(self, time_elapsed, epochs, telegram=None):

        if epochs is None:
            message = "Finalization finished in " + time_elapsed
        else:
            message = "Finalization finished after " + str(epochs) + " epochs in " + time_elapsed

        self.log_line(message=message, telegram=telegram)

    def log_evaluation(self, time_elapsed, log_path_evaluation, test_accuracy, test_precision, test_recall,
                       test_f1_score, test_cohen_kappa_score, test_matthews_correlation_coefficient, telegram=None):

        message = "Evaluation finished after in " + time_elapsed + " with" + \
                  "\n* accuracy " + str(round(test_accuracy, 2)) + \
                  "\n* precision " + str(round(test_precision, 2)) + \
                  "\n* recall " + str(round(test_recall, 2)) + \
                  "\n* f1 score " + str(round(test_f1_score, 2)) + \
                  "\n* cohen's kappa score " + str(round(test_cohen_kappa_score, 2)) + \
                  "\n* matthews correlation coefficient " + str(round(test_matthews_correlation_coefficient, 2))

        file_path = os.path.join(log_path_evaluation, "plots", "confusion_matrix.png")
        if os.path.exists(file_path):
            with open(file_path, "rb") as confusion_matrix_file:
                self.log_line(message=message, images=[confusion_matrix_file], telegram=telegram)

    def log_training_end(self, time_elapsed, telegram=None):

        message = "Training finished in " + time_elapsed

        self.log_line(message=message, telegram=telegram)
