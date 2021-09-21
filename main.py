import getopt
import os
import sys
from datetime import datetime

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
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
from sliding_window_data_splitter import SlidingWindowDataSplitter
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from data_normalizer import DataNormalizer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_slice_plotter import BikeActivitySlicePlotter
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
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    epochs = 50_000
    learning_rate = 0.001
    random_state = 0

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hcqtde:l:", ["help", "clean", "quiet", "transient", "dry-run", "epochs=", "learningrate="])
    except getopt.GetoptError:
        print("main.py --help --clean --quiet --transient --dry-run --epochs <epochs> --learningrate <learningrate>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            print("main.py -e <epochs>")
            sys.exit()
        elif opt in ("-c", "--clean"):
            clean = True
        elif opt in ("-q", "--quiet"):
            quiet = True
        elif opt in ("-t", "--transient"):
            transient = True
        elif opt in ("-d", "--dry-run"):
            epochs = 1
            clean = True
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--learningrate"):
            learning_rate = float(arg)

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

    # Initialize logger
    logger = LoggerFacade(log_path, console=True, file=True)
    logger.log_line("Start Training")

    # Print parameters
    logger.log_line("Parameters")
    logger.log_line("❄ starttime: " + str(start_time))
    logger.log_line("❄ clean: " + str(clean))
    logger.log_line("❄ quiet: " + str(quiet))
    logger.log_line("❄ epochs: " + str(epochs))
    logger.log_line("❄ learningrate: " + str(learning_rate))

    #
    # Data pre-processing
    #

    SlidingWindowDataSplitter().run(
        logger=logger,
        data_path=data_path + "/measurements/csv",
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

    filtered_dataframes = DataFilterer().run(logger=logger, dataframes=dataframes)

    BikeActivityPlotter().run(
        logger=logger,
        data_path=data_path + "/measurements/csv",
        results_path=log_path + "/plots/bike-activity",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=clean,
        quiet=quiet
    )

    # BikeActivitySlicePlotter().run(
    #     logger=logger,
    #     data_path=workspace_path + "/slices/raw",
    #     results_path=log_path + "/plots/bike-activity-sample",
    #     xlabel="time",
    #     ylabel="acceleration [m/sˆ2]/ speed [km/h]",
    #     clean=clean,
    #     quiet=quiet
    # )

    BikeActivitySurfaceTypePlotter().run(
        logger=logger,
        dataframes=filtered_dataframes,
        results_path=log_path + "/plots/bike-activity-surface-type",
        file_name="surface_type",
        title="Surface type distribution",
        description="Distribution of surface types in input data",
        xlabel="surface type",
        ylabel="percentage",
        clean=clean,
        quiet=quiet
    )

    train_dataframes, validation_dataframes, test_dataframes = TrainTestDataSplitter().run(
        logger=logger,
        dataframes=filtered_dataframes,
        test_size=0.15,
        random_state=random_state,
        quiet=True
    )

    BikeActivitySurfaceTypePlotter().run(
        logger=logger,
        dataframes=train_dataframes,
        results_path=log_path + "/plots/bike-activity-surface-type",
        file_name="surface_type_test",
        title="Surface type distribution (test)",
        description="Distribution of surface types in input data",
        xlabel="surface type",
        ylabel="percentage",
        clean=clean,
        quiet=quiet
    )

    BikeActivitySurfaceTypePlotter().run(
        logger=logger,
        dataframes=validation_dataframes,
        results_path=log_path + "/plots/bike-activity-surface-type",
        file_name="surface_type_validation",
        title="Surface type distribution (validation)",
        description="Distribution of surface types in input data",
        xlabel="surface type",
        ylabel="percentage",
        clean=clean,
        quiet=quiet
    )

    #
    # Data Preparation
    #

    dataframes = DataFilterer().run(logger=logger, dataframes=dataframes)
    dataframes = DataTransformer().run(logger=logger, dataframes=dataframes)
    dataframes = DataNormalizer().run(logger=logger, dataframes=dataframes)

    train_dataframes, validation_dataframes, test_dataframes = TrainTestDataSplitter().run(
        logger=logger,
        dataframes=dataframes,
        test_size=0.15,
        random_state=random_state,
        quiet=quiet
    )

    #
    # Modeling
    #

    CnnBaseModel().run(
        logger=logger,
        train_dataframes=train_dataframes,
        validation_dataframes=validation_dataframes,
        epochs=epochs,
        learning_rate=learning_rate,
        log_path=log_path,
        quiet=quiet
    )

    #
    # Evaluation
    #

    CnnBaseModel().evaluate(
        logger=logger,
        test_dataframes=test_dataframes,
        log_path=log_path,
        clean=clean,
        quiet=quiet
    )

    #
    #
    #

    if log_latest_path is not None:
        ResultCopier().copyDirectory(log_path, log_latest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
