import os
from pathlib import Path

import pandas as pd


#
# Main
#


class DataLoader:

    def run(self, logger, data_path):
        dataframes = {}

        for file_path in Path(data_path).rglob("*.csv"):
            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            dataframe = pd.read_csv(file_path, skiprows=1, names=[
                "bike_activity_uid", "bike_activity_sample_uid", "bike_activity_measurement", "bike_activity_measurement_timestamp",
                "bike_activity_measurement_lon", "bike_activity_measurement_lat", "bike_activity_measurement_speed",
                "bike_activity_measurement_accelerometer_x", "bike_activity_measurement_accelerometer_y",
                "bike_activity_measurement_accelerometer_z", "bike_activity_phone_position", "bike_activity_bike_type",
                "bike_activity_surface_type", "bike_activity_smoothness_type", "bike_activity_measurement_accelerometer"
            ])

            dataframes[file_base_name] = dataframe

        logger.log_line("Data loader finished with " + str(len(dataframes)) + " dataframes loaded")
        return dataframes
