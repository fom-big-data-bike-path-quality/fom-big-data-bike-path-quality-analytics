import getopt
import os
import sys

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/base_model'),
    os.path.join(os.getcwd(), 'lib/plotters'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from epoch_splitter import EpochSplitter
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_epoch_plotter import BikeActivityEpochPlotter
from train_test_data_splitter import TrainTestDataSplitter
from cnn_base_model_helper import CnnBaseModelHelper


#
# Main
#

def main(argv):
    # Set default values
    epochs = 3000
    learning_rate = 0.001

    # Read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hde:", ["dry-run", "epochs="])
    except getopt.GetoptError:
        print("main.py -e <epochs>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("main.py -e <epochs>")
            sys.exit()
        elif opt in ("-d", "--dry-run"):
            epochs = 1
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)

    # Print parameters
    print("Parameters")
    print("❄ epochs: " + str(epochs))

    # Set paths
    file_path = os.path.realpath(__file__)
    script_path = os.path.dirname(file_path)
    data_path = script_path + "/data/data"
    workspace_path = script_path + "/workspace"
    results_path = script_path + "/results"

    #
    # Data pre-processing
    #

    EpochSplitter().run(
        data_path=data_path + "/measurements/csv",
        results_path=workspace_path + "/epochs/raw",
        clean=True
    )

    dataframes = DataLoader().run(data_path=workspace_path + "/epochs/raw")

    #
    # Data Understanding
    #

    BikeActivityPlotter().run(
        data_path=data_path + "/measurements/csv",
        results_path=results_path + "/plots/bike-activity",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=True
    )

    BikeActivityEpochPlotter().run(
        data_path=workspace_path + "/epochs/raw",
        results_path=results_path + "/plots/bike-activity-sample",
        xlabel="time",
        ylabel="acceleration [m/sˆ2]/ speed [km/h]",
        clean=True
    )

    #
    # Data Preparation
    #

    random_state = 0

    dataframes = DataFilterer().run(dataframes)
    dataframes = DataTransformer().run(dataframes)

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
        n_epochs=epochs,
        learning_rate=learning_rate,
        workspace_path=workspace_path,
        results_path=results_path
    )


if __name__ == "__main__":
    main(sys.argv[1:])
