import numpy as np


class ModelLogger:

    def log_fold_results(self, logger, overall_validation_accuracy_history, overall_validation_precision_history,
                         overall_validation_recall_history, overall_validation_f1_score_history,
                         overall_validation_cohen_kappa_score_history,
                         overall_validation_matthew_correlation_coefficient_history, quiet):
        if not quiet:
            try:
                logger.log_line("Cross-validation metrics")
                logger.log_line(
                    "Mean accuracy " + str(round(np.mean(overall_validation_accuracy_history), 2)) + ", " +
                    "precision " + str(round(np.mean(overall_validation_precision_history), 2)) + ", " +
                    "recall " + str(round(np.mean(overall_validation_recall_history), 2)) + ", " +
                    "f1 score " + str(round(np.mean(overall_validation_f1_score_history), 2)) + ", " +
                    "cohen kappa score " + str(round(np.mean(overall_validation_cohen_kappa_score_history), 2)) + ", " +
                    "matthew correlation coefficient " + str(
                        round(np.mean(overall_validation_matthew_correlation_coefficient_history), 2))
                )
                logger.log_line(
                    "Standard deviation " + str(round(np.std(overall_validation_accuracy_history), 2)) + ", " +
                    "precision " + str(round(np.std(overall_validation_precision_history), 2)) + ", " +
                    "recall " + str(round(np.std(overall_validation_recall_history), 2)) + ", " +
                    "f1 score " + str(round(np.std(overall_validation_f1_score_history), 2)) + ", " +
                    "cohen kappa score " + str(round(np.std(overall_validation_cohen_kappa_score_history), 2)) + ", " +
                    "matthew correlation coefficient " + str(
                        round(np.std(overall_validation_matthew_correlation_coefficient_history), 2))
                )
            except:
                pass
