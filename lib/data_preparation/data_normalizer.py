import inspect

from sklearn.preprocessing import MinMaxScaler
from tracking_decorator import TrackingDecorator


#
# Main
#

class DataNormalizer:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, quiet=False):
        min_max_scaler = MinMaxScaler()

        for name, dataframe in list(dataframes.items()):
            dataframe["bike_activity_measurement_accelerometer"] = min_max_scaler.fit_transform(
                dataframe[['bike_activity_measurement_accelerometer']].values.astype(float))

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " normalized " + str(len(dataframes)) + " dataframes")

        return dataframes
