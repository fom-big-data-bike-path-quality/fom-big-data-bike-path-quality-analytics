import os
import sys

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib')
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from epoch_splitter import EpochSplitter
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from train_test_data_splitter import TrainTestDataSplitter

# Configuration

# Set script path
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

dataframes = DataFilterer().run(dataframes)
dataframes = DataTransformer().run(dataframes)

#
# Modeling
#

random_state = 0

train_dataframes, test_dataframes = TrainTestDataSplitter().run(dataframes=dataframes, test_size=0.15, random_state=random_state)

#
# Evaluation
#
