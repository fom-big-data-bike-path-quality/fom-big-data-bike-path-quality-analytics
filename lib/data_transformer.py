import math


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


#
# Main
#


class DataTransformer:

    def run(self, dataframes):
        for name, dataframe in list(dataframes.items()):
            dataframe["bike_activity_measurement_accelerometer"] = dataframe.apply(lambda row: getAccelerometer(row), axis=1)

        print("DataTransformer finished.")
        return dataframes
