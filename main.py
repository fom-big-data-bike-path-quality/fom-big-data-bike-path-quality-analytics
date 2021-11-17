#!/usr/bin/env python3

import getopt
import os
import sys
from datetime import datetime

import torch

file_path = os.path.realpath(__file__)
script_path = os.path.dirname(file_path)

# Make library available in path
library_paths = [
    os.path.join(script_path, 'lib'),
    os.path.join(script_path, 'lib', 'log'),
    os.path.join(script_path, 'lib', 'data_pre_processing'),
    os.path.join(script_path, 'lib', 'data_preparation'),
    os.path.join(script_path, 'lib', 'plotters'),
    os.path.join(script_path, 'lib', 'models'),
    os.path.join(script_path, 'lib', 'models', 'base_model_knn_dtw'),
    os.path.join(script_path, 'lib', 'models', 'base_model_cnn'),
    os.path.join(script_path, 'lib', 'models', 'base_model_cnn', 'layers'),
    os.path.join(script_path, 'lib', 'models', 'base_model_lstm'),
    os.path.join(script_path, 'lib', 'cloud'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from logger_facade import LoggerFacade
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from data_normalizer import DataNormalizer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_surface_type_plotter import BikeActivitySurfaceTypePlotter
from train_test_data_splitter import TrainTestDataSplitter
from data_resampler import DataResampler
from cnn_base_model import CnnBaseModel
from knn_dtw_base_model import KnnDtwBaseModel
from lstm_base_model import LstmBaseModel
from result_handler import ResultHandler
from tracking_decorator import TrackingDecorator


#
# Main
#

@TrackingDecorator.track_time
def main(argv):
    training_start_time = datetime.now()
    training_start_time_string = training_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    available_model_names = ["cnn", "lstm", "knn-dtw"]

    # Set default values
    clean = False
    quiet = False
    transient = False
    dry_run = False
    skip_data_understanding = False
    skip_validation = False

    limit = None

    window_step = 50
    down_sampling_factor = 3.0

    model_name = None
    k_folds = 10
    k_nearest_neighbors = 10
    epochs = 10_000
    learning_rate: float = 0.001
    patience = 500
    slice_width = 500
    dropout = 0.5
    lstm_hidden_dimension = 128
    lstm_layer_dimension = 3

    measurement_speed_limit = 5.0

    test_size: float = 0.15
    random_state = 0

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hcqtdw:m:f:k:e:l:p:s:",
                                   ["help", "clean", "quiet", "transient", "dry-run", "skip-data-understanding",
                                    "skip-validation", "window-step=", "down-sampling-factor=", "model=", "k-folds=",
                                    "k-nearest-neighbors=", "epochs=", "learning-rate=", "patience=", "slice-width=",
                                    "dropout=", "lstm-hidden-dimension=", "lstm-layer-dimension="])
    except getopt.GetoptError as error:
        print(argv)
        print(error)
        print(
            "main.py --help --clean --quiet --transient --dry-run --skip-data-understanding --skip-validation " +
            "--window-step <window-step> --down-sampling-factor <down-sampling-factor> --model <model>" +
            "--k-folds <k-folds> --k-nearest-neighbors <k-nearest-neighbors> --epochs <epochs> "
            "--learning-rate <learning-rate> --patience <patience> --slice-width <slice-width> --dropout <dropout>" +
            "--lstm-hidden-dimension <lstm-hidden-dimension> --lstm-layer-dimension <lstm-layer-dimension>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python main.py [OPTION]...")
            print("")
            print("--help                                             show this help")
            print("--clean                                            clean intermediate results before start")
            print("--quiet                                            do not log outputs")
            print("--transient                                        do not store results")
            print(
                "--dry-run                                          only run a limited training to make sure syntax is correct")
            print("--skip-data-understanding                          skip data understanding")
            print("--skip-validation                                  skip validation")
            print("--window-step <window-step>                        step size used for sliding window data splitter")
            print(
                "--down-sampling-factor <down-sampling-factor>      factor by which target classes are capped in comparison to smallest class")
            print("--model <model>                                    name of the model to use for training")
            print("--k-folds <k-folds>                                number of k-folds")
            print(
                "--k-nearest-neighbors <k-nearest-neighbors>        number of nearest neighbors to consider in kNN approach")
            print("--epochs <epochs>                                  number of epochs")
            print("--learning-rate <learning-rate>                    learning rate")
            print(
                "--patience <patience>                              number of epochs to wait for improvements before finishing training")
            print("--slice-width <slice-width>                        number of measurements per slice")
            print("--dropout <dropout>                                dropout percentage")
            print("--lstm-hidden-dimension <lstm-hidden-dimension>    hidden dimensions in LSTM")
            print("--lstm-layer-dimension <lstm-layer-dimension>      layer dimensions in LSTM")
            print("")
            print("Examples:")
            print("  python main.py -c -m cnn -e 3000 -l 0.001")
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
            limit = 100
            k_folds = 5
        elif opt in "--skip-data-understanding":
            skip_data_understanding = True
        elif opt in "--skip-validation":
            skip_validation = True
        elif opt in ("-w", "--window-step"):
            window_step = int(arg)
        elif opt in "--down-sampling-factor":
            down_sampling_factor = float(arg)
        elif opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("-f", "--k-folds"):
            k_folds = int(arg)
        elif opt in ("-k", "--k-nearest-neighbors"):
            k_nearest_neighbors = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
        elif opt in ("-p", "--patience"):
            patience = int(arg)
        elif opt in ("-s", "--slice-width"):
            slice_width = int(arg)
        elif opt in "--dropout":
            dropout = float(arg)
        elif opt in "--lstm-hidden-dimension":
            lstm_hidden_dimension = int(arg)
        elif opt in "--lstm-layer-dimension":
            lstm_layer_dimension = int(arg)

    if not model_name in available_model_names:
        raise getopt.GetoptError("invalid model name. valid options are " + str(available_model_names))

    # Set paths
    data_path = os.path.join(script_path, "data", "data")
    slices_path = os.path.join(data_path, "measurements", "slices",
                               "width" + str(slice_width) + "_step" + str(window_step))

    # Set device name
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if not transient:
        log_path = os.path.join(script_path, "results", "results", model_name, training_start_time_string)
        log_latest_path = os.path.join(script_path, "results", "results", model_name, "latest")
    else:
        log_path = os.path.join(script_path, "results", "results", model_name, "transient")
        log_latest_path = None

    log_path_data_understanding = os.path.join(log_path, "02-data-understanding")
    log_path_modelling = os.path.join(log_path, "04-modelling")
    log_path_evaluation = os.path.join(log_path, "05-evaluation")

    # Initialize logger
    logger = LoggerFacade(log_path, console=True, file=True)

    logger.log_training_start(
        device_name=device_name,
        training_start_time_string=training_start_time_string,
        clean=clean,
        quiet=quiet,
        transient=transient,
        dry_run=dry_run,
        skip_data_understanding=skip_data_understanding,
        skip_validation=skip_validation,

        window_step=window_step,
        down_sampling_factor=down_sampling_factor,

        model_name=model_name,
        k_folds=k_folds,
        k_nearest_neighbors=k_nearest_neighbors,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        slice_width=slice_width,
        dropout=dropout,
        lstm_hidden_dimension=lstm_hidden_dimension,
        lstm_layer_dimension=lstm_layer_dimension,

        measurement_speed_limit=measurement_speed_limit,
        test_size=test_size,
        random_state=random_state,

        telegram=not quiet and not dry_run
    )

    dataframes = DataLoader().run(
        logger=logger,
        data_path=slices_path,
        limit=limit,
        quiet=quiet
    )

    #
    # Data Understanding
    #

    if not skip_data_understanding:
        filtered_dataframes = DataFilterer().run(logger=logger, dataframes=dataframes, slice_width=slice_width,
                                                 measurement_speed_limit=measurement_speed_limit, quiet=True)

        BikeActivityPlotter().run(
            logger=logger,
            data_path=os.path.join(data_path, "measurements", "csv"),
            results_path=os.path.join(log_path_data_understanding, "plots", "bike-activity"),
            xlabel="time",
            ylabel="acceleration [m/sˆ2]/ speed [km/h]",
            clean=clean,
            quiet=quiet
        )

        # BikeActivitySlicePlotter().run(
        #     logger=logger,
        #     data_path=slices_path,
        #     results_path=os.path.join(log_path_data_understanding, "plots", "bike-activity-sample"),
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

        resampled_train_dataframes = DataResampler().run_down_sampling(
            logger=logger,
            dataframes=train_dataframes,
            down_sampling_factor=down_sampling_factor,
            quiet=quiet
        )

        BikeActivitySurfaceTypePlotter().run(
            logger=logger,
            dataframes=train_dataframes,
            slice_width=slice_width,
            results_path=os.path.join(log_path_data_understanding, "plots", "bike-activity-surface-type"),
            file_name="surface_type_train",
            title="Surface type distribution (train)",
            description="Distribution of surface types in input data",
            xlabel="surface type",
            clean=clean,
            quiet=quiet
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
            clean=clean,
            quiet=quiet
        )

        BikeActivitySurfaceTypePlotter().run(
            logger=logger,
            dataframes=test_dataframes,
            slice_width=slice_width,
            results_path=os.path.join(log_path_data_understanding, "plots", "bike-activity-surface-type"),
            file_name="surface_type_test",
            title="Surface type distribution (test)",
            description="Distribution of surface types in input data",
            xlabel="surface type",
            clean=clean,
            quiet=quiet
        )

        BikeActivitySurfaceTypePlotter().run(
            logger=logger,
            dataframes=resampled_train_dataframes,
            slice_width=slice_width,
            results_path=log_path_data_understanding + "/plots/bike-activity-surface-type",
            file_name="surface_type_train_resampled",
            title="Surface type distribution (train, resampled)",
            description="Distribution of surface types in input data",
            xlabel="surface type",
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

    resampled_train_dataframes = DataResampler().run_down_sampling(
        logger=logger,
        dataframes=train_dataframes,
        down_sampling_factor=down_sampling_factor,
        run_after_label_encoding=True,
        quiet=quiet
    )

    #
    # Modeling
    #

    logger.log_modelling_start(
        model_name=model_name,
        train_dataframes=train_dataframes,
        resampled_train_dataframes=resampled_train_dataframes,
        test_dataframes=test_dataframes,
        telegram=not quiet and not dry_run
    )

    #
    # Model Initialization
    #

    if model_name == "knn-dtw":

        model = KnnDtwBaseModel(
            logger=logger,
            log_path_modelling=log_path_modelling,
            log_path_evaluation=log_path_evaluation,
            train_dataframes=train_dataframes,
            test_dataframes=test_dataframes,
            k_nearest_neighbors=k_nearest_neighbors,
            slice_width=slice_width
        )

    elif model_name == "cnn":

        model = CnnBaseModel(
            logger=logger,
            log_path_modelling=log_path_modelling,
            log_path_evaluation=log_path_evaluation,
            train_dataframes=train_dataframes,
            test_dataframes=test_dataframes,

            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            dropout=dropout,
            slice_width=slice_width,
        )

    elif model_name == "lstm":

        model = LstmBaseModel(
            logger=logger,
            log_path_modelling=log_path_modelling,
            log_path_evaluation=log_path_evaluation,
            train_dataframes=train_dataframes,
            test_dataframes=test_dataframes,

            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            dropout=dropout,
            slice_width=slice_width,

            lstm_hidden_dimension=lstm_hidden_dimension,
            lstm_layer_dimension=lstm_layer_dimension
        )

    #
    # Validation
    #

    if not skip_validation:
        finalize_epochs = model.validate(
            k_folds=k_folds,

            random_state=random_state,
            quiet=quiet,
            dry_run=dry_run
        )
    else:
        finalize_epochs = epochs

    model.finalize(
        epochs=finalize_epochs,
        quiet=quiet,
        dry_run=dry_run
    )

    #
    # Evaluation
    #

    model.evaluate(
        clean=clean,
        quiet=quiet,
        dry_run=dry_run
    )

    #
    #
    #

    if not transient:
        ResultHandler().copy_directory(
            source_dir=log_path,
            destination_dir=log_latest_path
        )
        ResultHandler().zip_directory(
            source_dir=log_path,
            destination_dir=os.path.join(script_path, "results", "results"),
            zip_name=training_start_time_string + ".zip",
            zip_root_dir=training_start_time_string
        )
        ResultHandler().upload_results(
            logger=logger,
            upload_file_path=os.path.join(script_path, "results", "results", training_start_time_string + ".zip"),
            project_id="bike-path-quality",
            bucket_name="bike-path-quality-results",
            quiet=quiet
        )

    logger.log_training_end(
        time_elapsed="{}".format(datetime.now() - training_start_time),
        telegram=not quiet and not dry_run
    )


if __name__ == "__main__":
    main(sys.argv[1:])
