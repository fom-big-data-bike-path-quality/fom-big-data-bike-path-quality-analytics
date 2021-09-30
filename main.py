import getopt
import os
import sys
from datetime import datetime

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/statistics'),
    os.path.join(os.getcwd(), 'lib/data_pre_processing'),
    os.path.join(os.getcwd(), 'lib/data_preparation'),
    os.path.join(os.getcwd(), 'lib/log'),
    os.path.join(os.getcwd(), 'lib/plotters'),
    os.path.join(os.getcwd(), 'lib/base_model'),
    os.path.join(os.getcwd(), 'lib/base_model/layers'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from logger_facade import LoggerFacade
from telegram_logger import TelegramLogger
from input_data_statistics import InputDataStatistics
from sliding_window_data_splitter import SlidingWindowDataSplitter
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from data_normalizer import DataNormalizer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_surface_type_plotter import BikeActivitySurfaceTypePlotter
from train_test_data_splitter import TrainTestDataSplitter
from cnn_base_model import CnnBaseModel
from result_copier import ResultCopier
from tracking_decorator import TrackingDecorator


#
# Main
#

@TrackingDecorator.track_time
def main(argv):
    # Set default values
    clean = False
    quiet = False
    transient = False
    dry_run = False
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    k_folds = 10
    epochs = 50_000
    learning_rate = 0.001
    patience = 500
    slice_width = 500
    window_step = 20

    measurement_speed_limit = 5.0

    test_size = 0.15
    random_state = 0

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hcqtdke:l:p:s:w:", ["help", "clean", "quiet", "transient", "dry-run", "kfolds=", "epochs=",
                                                         "learningrate=", "patience=", "slicewidth=", "windowstep="])
    except getopt.GetoptError:
        print("main.py --help --clean --quiet --transient --dry-run --kfolds <k-folds> --epochs <epochs> --learning-rate <learning-rate> "
              "--patience <patience> --slice-width <slice-width> --window-step <windowstep>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("main.py")
            print("--help                           show this help")
            print("--clean                          clean intermediate results before start")
            print("--quiet                          do not log outputs")
            print("--transient                      do not store results")
            print("--dry-run                        only run a limited training to make sure syntax is correct")
            print("--k-folds <k-folds>              number of k-folds")
            print("--epochs <epochs>                number of epochs")
            print("--learning-rate <learning-rate>  learning rate")
            print("--patience <patience>            number of epochs to wait for improvements before finishing training")
            print("--slice-width <slice-width>      number of measurements per slice")
            print("--window-step <window-step>      step size used for sliding window data splitter")
            sys.exit()
        elif opt in ("-c", "--clean"):
            clean = True
        elif opt in ("-q", "--quiet"):
            quiet = True
        elif opt in ("-t", "--transient"):
            transient = True
        elif opt in ("-d", "--dry-run"):
            epochs = 2
            clean = True
            transient = True
            dry_run = True
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-k", "--kfolds"):
            k_folds = int(arg)
        elif opt in ("-l", "--learningrate"):
            learning_rate = float(arg)
        elif opt in ("-p", "--patience"):
            patience = int(arg)
        elif opt in ("-s", "--slicewidth"):
            slice_width = int(arg)
        elif opt in ("-w", "--windowstep"):
            window_step = int(arg)

    # Set paths
    file_path = os.path.realpath(__file__)
    script_path = os.path.dirname(file_path)
    data_path = os.path.join(script_path, "data/data")
    workspace_path = os.path.join(script_path, "workspace")

    if not transient:
        log_path = os.path.join(script_path, "models", "models", start_time)
        log_latest_path = os.path.join(script_path, "models", "models", "latest")
    else:
        log_path = os.path.join(script_path, "models", "models", "transient")
        log_latest_path = None

    log_path_data_understanding = os.path.join(log_path, "02-data-understanding")
    log_path_modelling = os.path.join(log_path, "04-modelling")
    log_path_evaluation = os.path.join(log_path, "05-evaluation")

    # Initialize logger
    logger = LoggerFacade(log_path, console=True, file=True)
    logger.log_line("Start Training")

    # Print parameters
    logger.log_line("Parameters")
    logger.log_line("❄ start time: " + str(start_time))
    logger.log_line("❄ clean: " + str(clean))
    logger.log_line("❄ quiet: " + str(quiet))
    logger.log_line("❄ k-folds: " + str(k_folds))
    logger.log_line("❄ epochs: " + str(epochs))
    logger.log_line("❄ learning rate: " + str(learning_rate))
    logger.log_line("❄ patience: " + str(patience))
    logger.log_line("❄ slice width: " + str(slice_width))
    logger.log_line("❄ window step: " + str(window_step))
    logger.log_line("❄ test size: " + str(test_size))

    #
    # Statistics
    #

    InputDataStatistics().run(
        logger=logger,
        data_path=data_path + "/measurements/csv",
        measurement_speed_limit=measurement_speed_limit,
        clean=clean,
        quiet=quiet
    )

    #
    # Data pre-processing
    #

    SlidingWindowDataSplitter().run(
        logger=logger,
        data_path=data_path + "/measurements/csv",
        slice_width=slice_width,
        window_step=window_step,
        results_path=workspace_path + "/slices/raw",
        clean=clean,
        quiet=quiet
    )

    dataframes = DataLoader().run(
        logger=logger,
        data_path=workspace_path + "/slices/raw",
        quiet=quiet
    )

    #
    # Data Understanding
    #

    filtered_dataframes = DataFilterer().run(logger=logger, dataframes=dataframes, slice_width=slice_width,
                                             measurement_speed_limit=measurement_speed_limit, quiet=True)

    BikeActivityPlotter().run(
        logger=logger,
        data_path=data_path + "/measurements/csv",
        results_path=log_path_data_understanding + "/plots/bike-activity",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=clean,
        quiet=quiet
    )

    # BikeActivitySlicePlotter().run(
    #     logger=logger,
    #     data_path=workspace_path + "/slices/raw",
    #     results_path=log_path_data_understanding + "/plots/bike-activity-sample",
    #     xlabel="time",
    #     ylabel="acceleration [m/sˆ2]/ speed [km/h]",
    #     clean=clean,
    #     quiet=quiet
    # )

    train_dataframes, test_dataframes = TrainTestDataSplitter().run(
        logger=logger,
        dataframes=filtered_dataframes,
        test_size=test_size,
        random_state=random_state,
        quiet=True
    )

    BikeActivitySurfaceTypePlotter().run(
        logger=logger,
        dataframes=train_dataframes,
        slice_width=slice_width,
        results_path=log_path_data_understanding + "/plots/bike-activity-surface-type",
        file_name="surface_type_train",
        title="Surface type distribution (train)",
        description="Distribution of surface types in input data",
        xlabel="surface type",
        ylabel="percentage",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run(
        logger=logger,
        dataframes=test_dataframes,
        slice_width=slice_width,
        results_path=log_path_data_understanding + "/plots/bike-activity-surface-type",
        file_name="surface_type_test",
        title="Surface type distribution (test)",
        description="Distribution of surface types in input data",
        xlabel="surface type",
        ylabel="percentage",
        clean=clean,
        quiet=quiet
    )

    #
    # Data Preparation
    #

    dataframes = DataFilterer().run(logger=logger, dataframes=dataframes, slice_width=slice_width,
                                    measurement_speed_limit=measurement_speed_limit, quiet=quiet)
    dataframes = DataTransformer().run(logger=logger, dataframes=dataframes, quiet=quiet)
    dataframes = DataNormalizer().run(logger=logger, dataframes=dataframes, quiet=quiet)

    train_dataframes, test_dataframes = TrainTestDataSplitter().run(
        logger=logger,
        dataframes=dataframes,
        test_size=test_size,
        random_state=random_state,
        quiet=quiet
    )

    #
    # Modeling
    #

    finalize_epochs = CnnBaseModel().validate(
        logger=logger,
        dataframes=train_dataframes,
        k_folds=k_folds,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        slice_width=slice_width,
        log_path=log_path_modelling,
        quiet=quiet
    )

    CnnBaseModel().finalize(
        logger=logger,
        dataframes=train_dataframes,
        epochs=finalize_epochs,
        learning_rate=learning_rate,
        slice_width=slice_width,
        log_path=log_path,
        quiet=quiet
    )

    #
    # Evaluation
    #

    test_accuracy, \
    test_precision, \
    test_recall, \
    test_f1_score, \
    test_cohen_kappa_score, \
    test_matthew_correlation_coefficient = CnnBaseModel().evaluate(
        logger=logger,
        dataframes=test_dataframes,
        slice_width=slice_width,
        model_path=log_path_modelling,
        log_path=log_path_evaluation,
        clean=clean,
        quiet=quiet
    )

    #
    #
    #

    ResultCopier().copyDirectory(log_path, log_latest_path)

    if not quiet and not dry_run:
        TelegramLogger().log_results(
            logger=logger,
            log_path_modelling=log_path_modelling,
            log_path_evaluation=log_path_evaluation,

            k_folds=k_folds,
            epochs=epochs,
            finalize_epochs=finalize_epochs,
            learning_rate=learning_rate,
            patience=patience,
            slice_width=slice_width,
            window_step=window_step,
            measurement_speed_limit=measurement_speed_limit,
            test_size=test_size,
            random_state=random_state,

            test_accuracy=test_accuracy,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1_score=test_f1_score,
            test_cohen_kappa_score=test_cohen_kappa_score,
            test_matthew_correlation_coefficient=test_matthew_correlation_coefficient
        )


if __name__ == "__main__":
    main(sys.argv[1:])
