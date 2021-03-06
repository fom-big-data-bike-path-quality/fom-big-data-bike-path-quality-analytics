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
    os.path.join(script_path, 'lib', 'data_statistics'),
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
from data_statistics import DataStatistics
from data_transformer import DataTransformer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_slice_plotter import BikeActivitySlicePlotter
from bike_activity_surface_type_plotter import BikeActivitySurfaceTypePlotter
from sliding_window_train_test_data_splitter import SlidingWindowTrainTestDataSplitter
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

    slice_width = 500
    window_step = 500

    measurement_speed_limit = 5.0
    down_sampling_factor = 1.5
    test_size: float = 0.2

    random_state = 0

    model_name = None
    k_folds = 10
    k_nearest_neighbors = 10
    subsample_step = 1
    max_warping_window = 500
    epochs = 10_000
    batch_size = 128
    patience = 1_000
    learning_rate: float = 0.001
    dropout = 0.5
    lstm_hidden_dimension = 128
    lstm_layer_dimension = 3

    gcp_project_id = "bike-path-quality-339900"
    gcp_bucket_name = "bike-path-quality-training-results"
    gcp_token_file = "bike-path-quality-339900-a8e468a52c18.json"

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hcqtds:w:m:f:k:e:b:l:p:",
                                   ["help", "clean", "quiet", "transient", "dry-run", "skip-data-understanding",
                                    "skip-validation", "slice-width=", "window-step=", "down-sampling-factor=",
                                    "model=", "k-folds=", "k-nearest-neighbors=", "dtw-subsample-step=",
                                    "dtw-max-warping-window=", "epochs=",  "batch-size=", "learning-rate=", "patience=",
                                    "dropout=", "lstm-hidden-dimension=", "lstm-layer-dimension="])
    except getopt.GetoptError as error:
        print(argv)
        print(error)
        print(
            "main.py --help --clean --quiet --transient --dry-run --skip-data-understanding --skip-validation "
            "--slice-width <slice-width> --window-step <window-step> --down-sampling-factor <down-sampling-factor> "
            "--model <model> --k-folds <k-folds> --k-nearest-neighbors <k-nearest-neighbors> "
            "--dtw-subsample-step <dtw-subsample-step> --dtw-max-warping-window <dtw-max-warping-window> "
            "--epochs <epochs> --batch_size <batch_size> --patience <patience> --learning-rate <learning-rate>"
            "--dropout <dropout> --lstm-hidden-dimension <lstm-hidden-dimension> "
            "--lstm-layer-dimension <lstm-layer-dimension>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python main.py [OPTION]...")
            print("")
            print("-h, --help                                         show this help")
            print("-c, --clean                                        clean intermediate results before start")
            print("-q, --quiet                                        do not log outputs")
            print("-t, --transient                                    do not store results")
            print("-d, --dry-run                                      " +
                  "only run a limited training to make sure syntax is correct")
            print("")
            print("--skip-data-understanding                          skip data understanding")
            print("--skip-validation                                  skip validation")
            print("")
            print("-s, --slice-width <slice-width>                    number of measurements per slice")
            print("-w, --window-step <window-step>                    step size used for sliding window data splitter")
            print("--down-sampling-factor <down-sampling-factor>      " +
                  "factor by which target classes are capped in comparison to smallest class")
            print("-m, --model <model>                                name of the model to use for training")
            print("-f, --k-folds <k-folds>                            number of k-folds")
            print("")
            print("-k, --k-nearest-neighbors <k-nearest-neighbors>    " +
                  "number of nearest neighbors to consider in kNN approach")
            print("--dtw-subsample-step <dtw-subsample-step>          subsample steps for DTW")
            print("--dtw-max-warping-window <dtw-max-warping-window>  max warping window for DTW")
            print("")
            print("-e, --epochs <epochs>                              number of epochs")
            print("-b, --batch-size <batch-size>                      batch size")
            print("-p, --patience <patience>                          " +
                  "number of epochs to wait for improvements before finishing training")
            print("-l, --learning-rate <learning-rate>                learning rate")
            print("--dropout <dropout>                                dropout percentage")
            print("--lstm-hidden-dimension <lstm-hidden-dimension>    hidden dimensions in LSTM")
            print("--lstm-layer-dimension <lstm-layer-dimension>      layer dimensions in LSTM")
            print("")
            print("Examples:")
            print("  python main.py -c -m knn-dtw -k 10")
            print("  python main.py -c -m lstm -s 500 -w 500 --lstm-hidden-dimension 128 --lstm-layer-dimension 3")
            print("  python main.py -c -m cnn -s 500 -w 500 ")
            sys.exit()
        elif opt in ("-c", "--clean"):
            clean = True
        elif opt in ("-q", "--quiet"):
            quiet = True
        elif opt in ("-t", "--transient"):
            transient = True
        elif opt in ("-d", "--dry-run"):
            clean = True
            transient = True
            dry_run = True
            limit = 10
            window_step = 500
            slice_width = 500
            epochs = 2
            k_folds = 5
        elif opt in "--skip-data-understanding":
            skip_data_understanding = True
        elif opt in "--skip-validation":
            skip_validation = True
        elif opt in ("-s", "--slice-width"):
            slice_width = int(arg)
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
        elif opt in "--dtw-subsample-step":
            subsample_step = int(arg)
        elif opt in "--dtw-max-warping-window":
            max_warping_window = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-p", "--patience"):
            patience = int(arg)
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
        elif opt in "--dropout":
            dropout = float(arg)
        elif opt in "--lstm-hidden-dimension":
            lstm_hidden_dimension = int(arg)
        elif opt in "--lstm-layer-dimension":
            lstm_layer_dimension = int(arg)

    if model_name not in available_model_names:
        raise getopt.GetoptError("invalid model name. valid options are " + str(available_model_names))

    # Set paths
    data_path = os.path.join(script_path, "data", "data")
    raw_data_path = os.path.join(data_path, "measurements", "csv")
    slices_path = os.path.join(data_path, "measurements", "slices")

    # Set device name
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if not transient:
        log_path = os.path.join(script_path, "results", "results", model_name, training_start_time_string)
        log_latest_path = os.path.join(script_path, "results", "results", model_name, "latest")
    else:
        log_path = os.path.join(script_path, "results", "results", model_name, "transient")
        log_latest_path = None

    log_path_data_understanding = os.path.join(log_path, "02-data-understanding")
    log_path_data_preparation = os.path.join(log_path, "03-data-preparation")
    log_path_modelling = os.path.join(log_path, "04-modelling")
    log_path_evaluation = os.path.join(log_path, "05-evaluation")

    # Initialize logger
    logger = LoggerFacade(log_path, console=True, file=True)

    logger.log_training_start(
        opts=opts,
        device_name=device_name,
        training_start_time_string=training_start_time_string,
        clean=clean,
        quiet=quiet,
        transient=transient,
        dry_run=dry_run,
        skip_data_understanding=skip_data_understanding,
        skip_validation=skip_validation,

        slice_width=slice_width,
        window_step=window_step,

        down_sampling_factor=down_sampling_factor,
        measurement_speed_limit=measurement_speed_limit,
        test_size=test_size,
        random_state=random_state,

        model_name=model_name,
        k_folds=k_folds,
        k_nearest_neighbors=k_nearest_neighbors,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,

        dropout=dropout,
        lstm_hidden_dimension=lstm_hidden_dimension,
        lstm_layer_dimension=lstm_layer_dimension,

        telegram=not quiet and not dry_run
    )

    dataframes = DataLoader().run(
        logger=logger,
        data_path=raw_data_path,
        limit=limit,
        quiet=quiet
    )

    #
    # Data Understanding
    #

    if not skip_data_understanding:
        BikeActivityPlotter().run(
            logger=logger,
            data_path=os.path.join(data_path, "measurements", "csv"),
            results_path=os.path.join(log_path_data_understanding, "bike-activity"),
            xlabel="time",
            ylabel="acceleration [m/s??2]/ speed [km/h]",
            clean=clean,
            quiet=quiet
        )

        BikeActivitySlicePlotter().run(
            logger=logger,
            data_path=slices_path,
            results_path=os.path.join(log_path_data_understanding, "bike-activity-sample"),
            xlabel="time",
            ylabel="acceleration [m/s??2]/ speed [km/h]",
            clean=clean,
            quiet=quiet
        )

    #
    # Data Preparation
    #

    train_dataframes, test_dataframes = SlidingWindowTrainTestDataSplitter().run(
        logger=logger,
        dataframes=dataframes,
        test_size=test_size,
        slice_width=slice_width,
        window_step=window_step,
        quiet=quiet
    )

    train_dataframes = DataFilterer().run(logger=logger, dataframes=train_dataframes, slice_width=slice_width,
                                          measurement_speed_limit=measurement_speed_limit,
                                          keep_unflagged_lab_conditions=False, quiet=quiet)
    train_dataframes = DataTransformer().run(logger=logger, dataframes=train_dataframes, quiet=quiet)
    # train_dataframes = DataNormalizer().run(logger=logger, dataframes=train_dataframes, quiet=quiet)

    test_dataframes = DataFilterer().run(logger=logger, dataframes=test_dataframes, slice_width=slice_width,
                                         measurement_speed_limit=measurement_speed_limit,
                                         keep_unflagged_lab_conditions=False, quiet=quiet)
    test_dataframes = DataTransformer().run(logger=logger, dataframes=test_dataframes, quiet=quiet)
    # test_dataframes = DataNormalizer().run(logger=logger, dataframes=test_dataframes, quiet=quiet)

    resampled_train_dataframes = DataResampler().run_down_sampling(
        logger=logger,
        dataframes=train_dataframes,
        down_sampling_factor=down_sampling_factor,
        run_after_label_encoding=True,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar(
        logger=logger,
        data=DataStatistics().run(
            dataframes=train_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_train",
        title="Surface type distribution (train)",
        description="Distribution of surface types in train data",
        xlabel="surface type",
        color="#3A6FB0",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar(
        logger=logger,
        data=DataStatistics().run(
            dataframes=test_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_test",
        title="Surface type distribution (test)",
        description="Distribution of surface types in test data",
        xlabel="surface type",
        color="#79ABD1",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar(
        logger=logger,
        data=DataStatistics().run(
            dataframes=train_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_train",
        title="Surface type distribution (train)",
        description="Distribution of surface types in train data",
        xlabel="surface type",
        color="#3A6FB0",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar2(
        logger=logger,
        data1=DataStatistics().run(
            dataframes=train_dataframes
        ),
        data2=DataStatistics().run(
            dataframes=test_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_train_test",
        title="Surface type distribution (train, test)",
        description="Distribution of surface types in train and test data",
        xlabel="surface type",
        label1="train",
        label2="test",
        color1="#3A6FB0",
        color2="#79ABD1",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar(
        logger=logger,
        data=DataStatistics().run(
            dataframes=resampled_train_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_train_resampled",
        title="Surface type distribution (train, resampled)",
        description="Distribution of surface types in resampled train data",
        xlabel="surface type",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run_bar2(
        logger=logger,
        data1=DataStatistics().run(
            dataframes=train_dataframes
        ),
        data2=DataStatistics().run(
            dataframes=resampled_train_dataframes
        ),
        results_path=log_path_data_preparation,
        file_name="surface_type_train_train_resampled",
        title="Surface type distribution (train, train resampled)",
        description="Distribution of surface types in train and resampled data",
        xlabel="surface type",
        label1="train",
        label2="train resampled",
        color1="#3A6FB0",
        color2="#79ABD1",
        clean=clean,
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

    model = None

    if model_name == "knn-dtw":

        model = KnnDtwBaseModel(
            logger=logger,
            log_path_modelling=log_path_modelling,
            log_path_evaluation=log_path_evaluation,
            train_dataframes=train_dataframes,
            test_dataframes=test_dataframes,
            k_nearest_neighbors=k_nearest_neighbors,
            subsample_step=subsample_step,
            max_warping_window=max_warping_window,
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
            batch_size=batch_size,
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
            batch_size=batch_size,
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
        average_epochs = model.validate(
            k_folds=k_folds,

            random_state=random_state,
            quiet=quiet,
            dry_run=dry_run
        )
    else:
        average_epochs = epochs

    model.finalize(
        epochs=epochs,
        average_epochs=average_epochs,
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
            gcp_token_file=gcp_token_file,
            upload_file_path=os.path.join(script_path, "results", "results", training_start_time_string + ".zip"),
            project_id=gcp_project_id,
            bucket_name=gcp_bucket_name,
            quiet=quiet
        )

    logger.log_training_end(
        time_elapsed="{}".format(datetime.now() - training_start_time),
        telegram=not quiet and not dry_run
    )


if __name__ == "__main__":
    main(sys.argv[1:])
