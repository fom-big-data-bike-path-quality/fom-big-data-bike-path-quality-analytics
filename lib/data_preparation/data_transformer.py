import inspect
import math
from tqdm import tqdm
import pandas as pd
from label_encoder import LabelEncoder
from tracking_decorator import TrackingDecorator


def getAccelerometer(row):
    """
    Calculates root mean square of accelerometer value components
    """

    bike_activity_measurement_accelerometer_x = float(row["bike_activity_measurement_accelerometer_x"])
    bike_activity_measurement_accelerometer_y = float(row["bike_activity_measurement_accelerometer_y"])
    bike_activity_measurement_accelerometer_z = float(row["bike_activity_measurement_accelerometer_z"])
    return math.sqrt((bike_activity_measurement_accelerometer_x ** 2
                      + bike_activity_measurement_accelerometer_y ** 2
                      + bike_activity_measurement_accelerometer_z ** 2) / 3)


def label_encoding(row):
    bike_activity_surface_type = row["bike_activity_surface_type"]

    return LabelEncoder().label_to_index(bike_activity_surface_type)

def reverse_label_encoding(index):
    return LabelEncoder().classes[index]

#
# Main
#

class DataTransformer:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, skip_label_encode_surface_type=False, quiet=False):

        copied_dataframes = dataframes.copy()

        progress_bar = tqdm(iterable=copied_dataframes.items(), unit="dataframe", desc="Transform dataframes")
        for name, dataframe in progress_bar:

            dataframe["bike_activity_measurement_accelerometer"] = pd.to_numeric(dataframe.apply(lambda row: getAccelerometer(row), axis=1))

            if not skip_label_encode_surface_type:
                dataframe["bike_activity_surface_type"] = dataframe.apply(lambda row: label_encoding(row), axis=1)
            else:
                dataframe["bike_activity_surface_type"] = dataframe.apply(lambda row: -1, axis=1)

            dataframe.drop(["bike_activity_uid",
                            "bike_activity_sample_uid",
                            "bike_activity_measurement",
                            "bike_activity_measurement_timestamp",
                            "bike_activity_measurement_lon",
                            "bike_activity_measurement_lat",
                            "bike_activity_measurement_speed",
                            "bike_activity_measurement_accelerometer_x",
                            "bike_activity_measurement_accelerometer_y",
                            "bike_activity_measurement_accelerometer_z",
                            "bike_activity_phone_position",
                            "bike_activity_bike_type",
                            "bike_activity_smoothness_type"], axis=1, inplace=True)

            if "bike_activity_flagged_lab_conditions" in dataframe.columns:
                dataframe.drop(["bike_activity_flagged_lab_conditions"], axis=1, inplace=True)

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " transformed " + str(len(copied_dataframes)) + " dataframes")

        return copied_dataframes

    def run_reverse(self, index):
        return reverse_label_encoding(index)