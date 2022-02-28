import inspect

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tracking_decorator import TrackingDecorator


#
# Main
#

class DataNormalizer:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, quiet=False):

        copied_dataframes = dataframes.copy()

        min_max_scaler = MinMaxScaler()

        progress_bar = tqdm(iterable=copied_dataframes.items(), unit="dataframe", desc="Normalize data frames")
        for name, dataframe in progress_bar:
            dataframe["bike_activity_measurement_accelerometer"] = min_max_scaler.fit_transform(
                dataframe[['bike_activity_measurement_accelerometer']].values.astype(float))

        progress_bar.close()

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(f"{class_name}.{function_name} normalized {str(len(dataframes))} dataframes")

        return copied_dataframes
