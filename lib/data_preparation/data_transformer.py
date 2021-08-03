import math
import pandas as pd


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


def getLabelEncoding(row):
    bike_activity_surface_type = row["bike_activity_surface_type"]

    if bike_activity_surface_type == "paved":
        return 0
    elif bike_activity_surface_type == "asphalt":
        return 1
    elif bike_activity_surface_type == "concrete":
        return 2
    elif bike_activity_surface_type == "concrete lanes":
        return 3
    elif bike_activity_surface_type == "concrete plates":
        return 4
    elif bike_activity_surface_type == "paving stones":
        return 5
    elif bike_activity_surface_type == "sett":
        return 6
    elif bike_activity_surface_type == "unhewn cobblestone":
        return 7
    elif bike_activity_surface_type == "cobblestone":
        return 8
    elif bike_activity_surface_type == "wood":
        return 9
    elif bike_activity_surface_type == "stepping_stones":
        return 10
    elif bike_activity_surface_type == "unpaved":
        return 11
    elif bike_activity_surface_type == "compacted":
        return 12
    elif bike_activity_surface_type == "fine gravel":
        return 13
    elif bike_activity_surface_type == "gravel":
        return 14
    elif bike_activity_surface_type == "rock":
        return 15
    elif bike_activity_surface_type == "pebblestone":
        return 16
    elif bike_activity_surface_type == "ground":
        return 17
    elif bike_activity_surface_type == "dirt":
        return 18
    elif bike_activity_surface_type == "ground":
        return 19
    elif bike_activity_surface_type == "earth":
        return 20
    elif bike_activity_surface_type == "grass":
        return 21
    elif bike_activity_surface_type == "mud":
        return 22
    elif bike_activity_surface_type == "sand":
        return 23
    elif bike_activity_surface_type == "woodchips":
        return 24
    elif bike_activity_surface_type == "snow":
        return 25
    elif bike_activity_surface_type == "ice":
        return 26
    elif bike_activity_surface_type == "salt":
        return 27
    else:
        return 99


#
# Main
#


class DataTransformer:

    def run(self, logger, dataframes):
        for name, dataframe in list(dataframes.items()):
            dataframe["bike_activity_measurement_accelerometer"] = pd.to_numeric(dataframe.apply(lambda row: getAccelerometer(row), axis=1))
            dataframe["bike_activity_surface_type"] = dataframe.apply(lambda row: getLabelEncoding(row), axis=1)

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

        logger.log_line("Data transformer finished with " + str(len(dataframes)) + " dataframes transformed")
        return dataframes
