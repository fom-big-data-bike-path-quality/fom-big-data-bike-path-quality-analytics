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
from data_transformer import DataTransformer
from bike_activity_plotter import BikeActivityPlotter
from bike_activity_sample_plotter import BikeActivitySamplePlotter

# Configuration

# Set script path
file_path = os.path.realpath(__file__)
script_path = os.path.dirname(file_path)

data_path = script_path + "/data/data"
workspace_path = script_path + "/workspace"
results_path = script_path + "/results"

#
# Data preparation
#

EpochSplitter().run(
    data_path=data_path + "/measurements/csv",
    results_path=workspace_path + "/epochs/raw",
)

DataTransformer().run(
    data_path=workspace_path + "/epochs/raw",
    results_path=workspace_path + "/epochs/transformed",
)

BikeActivitySamplePlotter().run(
    data_path=workspace_path + "/epochs/transformed",
    results_path=results_path + "/plots/bike-activity-sample",
    xlabel="time",
    ylabel="acceleration [m/sˆ2]/ speed [km/h]",
    clean=True
)

BikeActivityPlotter().run(
    data_path=data_path + "/measurements/csv",
    results_path=results_path + "/plots/bike-activity",
    xlabel="time",
    ylabel="acceleration [m/sˆ2]/ speed [km/h]",
    clean=True
)
