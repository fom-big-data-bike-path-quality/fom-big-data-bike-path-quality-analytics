import os

from bike_activity_surface_type_plotter import BikeActivitySurfaceTypePlotter
from training_result_plotter import TrainingResultPlotter


class ModelPlotter:

    def plot_fold_results(self, logger, log_path, fold_labels, overall_validation_accuracy_history,
                          overall_validation_precision_history, overall_validation_recall_history,
                          overall_validation_f1_score_history, overall_validation_cohen_kappa_score_history,
                          overall_validation_matthew_correlation_coefficient_history, quiet):
        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_accuracy_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-accuracy",
            title="Accuracy history",
            description="Accuracy history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_precision_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-precision",
            title="Precision history",
            description="Precision history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_recall_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-recall",
            title="Recall history",
            description="Recall history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_f1_score_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-f1-score",
            title="F1 score history",
            description="F1 score history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet)

        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_cohen_kappa_score_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-cohen-kappa-score",
            title="Cohen's kappa score history",
            description="Cohen's kappa score history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=overall_validation_matthew_correlation_coefficient_history,
            labels=fold_labels,
            results_path=os.path.join(log_path, "plots"),
            file_name="overall-matthew-correlation-coefficient-score",
            title="Matthews's correlation coefficient score history",
            description="Matthews's correlation coefficient score history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )

    def plot_fold_distribution(self, logger, log_path, train_dataframes, validation_dataframes, fold_index, slice_width,
                               quiet):
        BikeActivitySurfaceTypePlotter().run(
            logger=logger,
            dataframes=train_dataframes,
            slice_width=slice_width,
            results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
            file_name="surface_type_train",
            title="Surface type distribution (train)",
            description="Distribution of surface types in input data",
            xlabel="surface type",
            run_after_label_encoding=True,
            quiet=quiet
        )

        BikeActivitySurfaceTypePlotter().run(
            logger=logger,
            dataframes=validation_dataframes,
            slice_width=slice_width,
            results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
            file_name="surface_type_validation",
            title="Surface type distribution (validation)",
            description="Distribution of surface types in input data",
            xlabel="surface type",
            run_after_label_encoding=True,
            quiet=quiet
        )

    def plot_training_results(self, logger, log_path, train_loss_history, validation_loss_history,
                              train_accuracy_history, validation_accuracy_history, validation_precision_history,
                              validation_recall_history, validation_f1_score_history,
                              validation_cohen_kappa_score_history, validation_matthew_correlation_coefficient_history,
                              fold_index, quiet):
        TrainingResultPlotter().run(
            logger=logger,
            data=[train_loss_history, validation_loss_history],
            labels=["Train", "Validation"],
            results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
            file_name="loss",
            title="Loss history",
            description="Loss history",
            xlabel="Epoch",
            ylabel="Loss",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=[train_accuracy_history, validation_accuracy_history],
            labels=["Train", "Validation"],
            results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
            file_name="accuracy",
            title="Accuracy history",
            description="Accuracy history",
            xlabel="Epoch",
            ylabel="Accuracy",
            clean=True,
            quiet=quiet
        )

        TrainingResultPlotter().run(
            logger=logger,
            data=[validation_accuracy_history, validation_precision_history, validation_recall_history,
                  validation_f1_score_history, validation_cohen_kappa_score_history,
                  validation_matthew_correlation_coefficient_history],
            labels=["Accuracy", "Precision", "Recall", "F1 Score", "Cohen's Kappa Score",
                    "Matthew's Correlation Coefficient"],
            results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
            file_name="metrics",
            title="Metrics history",
            description="Metrics history",
            xlabel="Epoch",
            ylabel="Value",
            clean=True,
            quiet=quiet
        )