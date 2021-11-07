import inspect
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tracking_decorator import TrackingDecorator


#
# Main
#

class DataLoader:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, limit=100, quiet=False):
        dataframes = {}
        files = list(Path(data_path).rglob("*.csv"))

        if limit is not None:
            files = files[:limit]

        progress_bar = tqdm(iterable=files, unit="files", desc="Load data frames")
        for file_path in progress_bar:
            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            dataframe = pd.read_csv(file_path, skiprows=1, names=[
                "bike_activity_uid", "bike_activity_sample_uid", "bike_activity_measurement",
                "bike_activity_measurement_timestamp",
                "bike_activity_measurement_lon", "bike_activity_measurement_lat", "bike_activity_measurement_speed",
                "bike_activity_measurement_accelerometer_x", "bike_activity_measurement_accelerometer_y",
                "bike_activity_measurement_accelerometer_z", "bike_activity_phone_position", "bike_activity_bike_type",
                "bike_activity_flagged_lab_conditions", "bike_activity_surface_type", "bike_activity_smoothness_type",
                "bike_activity_measurement_accelerometer"
            ])

            dataframes[file_base_name] = dataframe

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(class_name + "." + function_name + " loaded " + str(len(dataframes)) + " dataframes")

        return dataframes
