from sklearn.preprocessing import MinMaxScaler


#
# Main
#


class DataNormalizer:

    def run(self, dataframes):
        min_max_scaler = MinMaxScaler()

        for name, dataframe in list(dataframes.items()):
            dataframe["bike_activity_measurement_accelerometer_scaled"] = min_max_scaler.fit_transform(
                dataframe[['bike_activity_measurement_accelerometer']].values.astype(float))
            dataframe.drop(["bike_activity_measurement_accelerometer"], axis=1, inplace=True)

        print("Data normalizer finished with " + str(len(dataframes)) + " dataframes normalized")
        return dataframes
