import inspect

from tqdm import tqdm
from tracking_decorator import TrackingDecorator

from label_encoder import LabelEncoder


def get_surface_type(dataframe, run_after_label_encoding):
    if run_after_label_encoding:
        surface_type_index = dataframe.bike_activity_surface_type.iloc[0]
        return LabelEncoder().classes[surface_type_index]
    else:
        return dataframe.bike_activity_surface_type.iloc[0]


#
# Main
#

class DataResampler:

    @TrackingDecorator.track_time
    def run_down_sampling(self, logger, dataframes, down_sampling_factor=3.0, run_after_label_encoding=False,
                          quiet=False):
        """
        Down-samples dataframes based on target class (bike_activity_surface_type)

        All classes will be capped to have a maximum size equal to target class size which is determined by
        multiplying the number of samples of the smallest class by a given class size factor.
        """

        copied_dataframes = dataframes.copy()

        # Surface types
        surface_types = LabelEncoder().classes

        # Cluster dataframes by surface type
        clustered_dataframes_pre_counter = {}
        clustered_dataframes_post_counter = {}
        for surface_type in surface_types:
            clustered_dataframes_pre_counter[surface_type] = 0
            clustered_dataframes_post_counter[surface_type] = 0

        # Count how many dataframes there are per target class
        for _, dataframe in list(copied_dataframes.items()):
            surface_type = get_surface_type(dataframe, run_after_label_encoding)
            if surface_type in surface_types:
                clustered_dataframes_pre_counter[surface_type] += 1

        # Determine size of classes
        min_class_size = None
        min_class_name = None
        for surface_type, class_size in list(clustered_dataframes_pre_counter.items()):
            if surface_type in surface_types and class_size > 0 and (
                    min_class_size is None or class_size < min_class_size):
                min_class_size = class_size
                min_class_name = surface_type

        # Define target class size
        target_class_size = int(min_class_size * down_sampling_factor)

        if not quiet:
            logger.log_line("smallest class is " + min_class_name + " with " + str(min_class_size) + " samples")
            logger.log_line("target class size is " + str(target_class_size))

        # Re-sample dataframes
        resampled_dataframes = {}

        progress_bar = tqdm(iterable=copied_dataframes.items(), unit="dataframe", desc="Re-sample data frames")
        for name, dataframe in progress_bar:
            surface_type = get_surface_type(dataframe, run_after_label_encoding)

            if surface_type in surface_types and clustered_dataframes_post_counter[surface_type] < target_class_size:
                resampled_dataframes[name] = dataframe
                clustered_dataframes_post_counter[surface_type] += 1

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            percentage = round(len(resampled_dataframes) / len(copied_dataframes), 2) * 100

            logger.log_line(
                class_name + "." + function_name + " kept " + str(len(resampled_dataframes)) + " of " +
                str(len(copied_dataframes)) + " dataframes after down-sampling (" + str(percentage) + "%)"
            )

        return resampled_dataframes
