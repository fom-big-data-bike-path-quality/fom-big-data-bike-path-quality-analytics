#
# Main
#


class DataFilterer:
    BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT = 5  # in km/h

    def run(self, dataframes):

        dataframes_count = len(dataframes.items())

        for name, dataframe in list(dataframes.items()):

            # Exclude dataframes which contain less than 500 measurements
            if len(dataframe) < 500:
                print("✗️ Filtering out " + name + " (less than 500 measurements)")
                dataframes.pop(name)
                continue

            # Exclude dataframes which contain surface type 'mixed'
            if "mixed" in dataframe.bike_activity_surface_type.values:
                print("✗️ Filtering out " + name + " (containing undefined surface type)")
                dataframes.pop(name)
                continue

            # Exclude dataframes which contain speeds slower than BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT
            if (dataframe.bike_activity_measurement_speed.values * 3.6 < self.BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT).any():
                print("✗️ Filtering out " + name + " (containing slow measurements)")
                dataframes.pop(name)
                continue

            print("✓️ Keeping " + name)

        dataframes_filtered_count = len(dataframes.items())

        print("Keeping " + str(dataframes_filtered_count) + "/" + str(dataframes_count)
              + " dataframes (" + str(round(dataframes_filtered_count / dataframes_count, 2) * 100) + "%)")
        print("DataFilterer finished.")
        return dataframes
