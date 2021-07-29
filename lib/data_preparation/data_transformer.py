import math
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

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
            dataframe["bike_activity_surface_type_label_encoded"] = le.fit_transform(dataframe["bike_activity_surface_type"])
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
                            "bike_activity_surface_type",
                            "bike_activity_smoothness_type"], axis=1, inplace=True)

        print("Data transformer finished with " + str(len(dataframes)) + " dataframes transformed")
        return dataframes
