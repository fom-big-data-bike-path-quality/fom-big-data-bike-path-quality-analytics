import getopt
import os
import sys

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/data_pre_processing'),
    os.path.join(os.getcwd(), 'lib/data_preparation'),
    os.path.join(os.getcwd(), 'lib/plotters'),
    os.path.join(os.getcwd(), 'lib/base_model'),
    os.path.join(os.getcwd(), 'lib/base_model/layers'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from data_splitter import DataSplitter
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from data_normalizer import DataNormalizer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_slice_plotter import BikeActivitySlicePlotter
from train_test_data_splitter import TrainTestDataSplitter
from cnn_base_model_helper import CnnBaseModelHelper


#
# Main
#

def main(argv):
    # Set default values
    clean = False
    epochs = 3000
    learning_rate = 0.001

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hcdel:", ["help", "clean", "dry-run", "epochs=", "learningrate="])
    except getopt.GetoptError:
        print("main.py -e <epochs>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            print("main.py -e <epochs>")
            sys.exit()
        elif opt in ("-c", "--clean"):
            clean = True
        elif opt in ("-d", "--dry-run"):
            epochs = 1
            clean = True
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--learningrate"):
            learning_rate = float(arg)

    # Print parameters
    print("Parameters")
    print("❄ epochs: " + str(epochs))
    print("❄ learningrate: " + str(learning_rate))

    # Set paths
    file_path = os.path.realpath(__file__)
    script_path = os.path.dirname(file_path)
    data_path = script_path + "/data/data"
    workspace_path = script_path + "/workspace"
    results_path = script_path + "/results"

    #
    # Data pre-processing
    #

    DataSplitter().run(
        data_path=data_path + "/measurements/csv",
        results_path=workspace_path + "/slices/raw",
        clean=clean
    )

    dataframes = DataLoader().run(
        data_path=workspace_path + "/slices/raw"
    )

    #
    # Data Understanding
    #

    BikeActivityPlotter().run(
        data_path=data_path + "/measurements/csv",
        results_path=results_path + "/plots/bike-activity",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=clean
    )

    BikeActivitySlicePlotter().run(
        data_path=workspace_path + "/slices/raw",
        results_path=results_path + "/plots/bike-activity-sample",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=clean
    )

    #
    # Data Preparation
    #

    random_state = 0

    dataframes = DataFilterer().run(dataframes)
    dataframes = DataTransformer().run(dataframes)
    dataframes = DataNormalizer().run(dataframes)

    train_dataframes, validation_dataframes, test_dataframes = TrainTestDataSplitter().run(
        dataframes=dataframes, test_size=0.15,
        random_state=random_state
    )

    #
    # Modeling
    #

    CnnBaseModelHelper().run(
        train_dataframes=train_dataframes,
        validation_dataframes=validation_dataframes,
        test_dataframes=test_dataframes,
        epochs=epochs,
        learning_rate=learning_rate,
        workspace_path=workspace_path,
        results_path=results_path
    )


if __name__ == "__main__":
    main(sys.argv[1:])
