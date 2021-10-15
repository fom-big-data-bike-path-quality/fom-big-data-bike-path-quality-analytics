import os
from pathlib import Path

import telegram_send


class TelegramLogger:

    def log_modelling_start(self, logger, k_folds, epochs, learning_rate, patience, slice_width,
                            window_step, measurement_speed_limit, test_size, random_state, train_dataframes,
                            test_dataframes):

        telegram_line = "🍱 Modelling with parameters " + \
                        "\n* k-folds " + str(k_folds) + \
                        "\n* epochs " + str(epochs) + \
                        "\n* learning rate " + str(learning_rate) + \
                        "\n* patience " + str(patience) + \
                        "\n* slice width " + str(slice_width) + \
                        "\n* window step " + str(window_step) + \
                        "\n* measurement speed limit " + str(measurement_speed_limit) + \
                        "\n* test size " + str(test_size) + \
                        "\n* random state " + str(random_state) + \
                        "\n* test_size " + str(random_state) + \
                        "\n\nstarted with" + \
                        "\n* train dataframes " + str(len(train_dataframes)) + \
                        "\n* test dataframes " + str(len(test_dataframes))

        # Set script path
        file_path = os.path.realpath(__file__)
        script_path = os.path.dirname(file_path)

        # Check for config file
        if not Path(os.path.join(script_path, "telegram.config")).exists():
            logger.log_line("✗️ Telegram config not found " + os.path.join(script_path, "telegram.config"))
            return

        # Send line to telegram
        telegram_send.send(
            messages=[telegram_line],
            parse_mode="html",
            conf=os.path.join(script_path, "telegram.config")
        )

    def log_fold(self, logger, time_elapsed, k_fold, k_folds, epochs, accuracy, precision, recall, f1_score,
                 cohen_kappa_score, matthew_correlation_coefficient):

        telegram_line = "🍱 Finished fold " + str(k_fold) + "/" + str(k_folds) + \
                        " after " + str(epochs) + " epochs in " + time_elapsed + \
                        "\n\nwith validation metrics" + \
                        "\n* accuracy " + str(accuracy) + \
                        "\n* precision " + str(precision) + \
                        "\n* recall " + str(recall) + \
                        "\n* f1_score " + str(f1_score) + \
                        "\n* cohen's kappa score " + str(cohen_kappa_score) + \
                        "\n* matthew's correlationcoefficient " + str(matthew_correlation_coefficient)

        # Set script path
        file_path = os.path.realpath(__file__)
        script_path = os.path.dirname(file_path)

        # Check for config file
        if not Path(os.path.join(script_path, "telegram.config")).exists():
            logger.log_line("✗️ Telegram config not found " + os.path.join(script_path, "telegram.config"))
            return

        # Send line to telegram
        telegram_send.send(
            messages=[telegram_line],
            parse_mode="html",
            conf=os.path.join(script_path, "telegram.config")
        )

    def log_modelling_end(self, logger, time_elapsed, log_path_modelling, log_path_evaluation, k_folds, epochs,
                          finalize_epochs, learning_rate, patience, slice_width, window_step, measurement_speed_limit,
                          test_size, random_state, test_accuracy, test_precision, test_recall,
                          test_f1_score, test_cohen_kappa_score, test_matthew_correlation_coefficient):

        # Retrieve image
        f1_score_path = os.path.join(log_path_modelling, "plots", "overall-f1-score.png")
        confusion_matrix_path = os.path.join(log_path_evaluation, "plots", "confusion_matrix.png")

        telegram_line = "🍱 Modelling with parameters " + \
                        "\n* k-folds " + str(k_folds) + \
                        "\n* epochs " + str(epochs) + \
                        "\n* learning rate " + str(learning_rate) + \
                        "\n* patience " + str(patience) + \
                        "\n* slice width " + str(slice_width) + \
                        "\n* window step " + str(window_step) + \
                        "\n* measurement speed limit " + str(measurement_speed_limit) + \
                        "\n* test size " + str(test_size) + \
                        "\n* random state " + str(random_state) + \
                        "\n\nfinished after " + str(finalize_epochs) + " epochs in " + time_elapsed + " with" + \
                        "\n* accuracy " + str(round(test_accuracy, 2)) + \
                        "\n* precision " + str(round(test_precision, 2)) + \
                        "\n* recall " + str(round(test_recall, 2)) + \
                        "\n* f1 score " + str(round(test_f1_score, 2)) + \
                        "\n* cohen's kappa score " + str(round(test_cohen_kappa_score, 2)) + \
                        "\n* matthew's correlation coefficient " + str(round(test_matthew_correlation_coefficient, 2))

        # Set script path
        file_path = os.path.realpath(__file__)
        script_path = os.path.dirname(file_path)

        # Check for config file
        if not Path(os.path.join(script_path, "telegram.config")).exists():
            logger.log_line("✗️ Telegram config not found " + os.path.join(script_path, "telegram.config"))
            return

        # Send line to telegram
        with open(f1_score_path, "rb") as f1_score_file, \
                open(confusion_matrix_path, "rb") as confusion_matrix_file:
            telegram_send.send(
                messages=[telegram_line],
                images=[f1_score_file, confusion_matrix_file],
                parse_mode="html",
                conf=os.path.join(script_path, "telegram.config")
            )
