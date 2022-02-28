import inspect
import math
from tqdm import tqdm
from tracking_decorator import TrackingDecorator


def slice_dataframe(name, dataframe, slice_width, window_step):
    slices = {}
    dataframe_size = dataframe.shape[0]

    # Set indices to first slice
    slice_index = 0
    start_index = 0
    end_index = slice_width

    # Iterate over data frame until
    while end_index < dataframe_size:
        slices[f"{name}-{str(slice_index).rjust(5, '0')}"] = dataframe.iloc[start_index:end_index]
        slice_index += 1
        start_index += window_step
        end_index += window_step

    return slices

#
# Main
#

class SlidingWindowTrainTestDataSplitter:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, test_size=0.2, slice_width=500, window_step=500, quiet=False):

        train_dataframes = {}
        test_dataframes = {}
        train_dataframes_dict = {}
        test_dataframes_dict = {}

        progress_bar = tqdm(iterable=dataframes.items(), unit="dataframe", desc="Split data frames")
        for name, dataframe in progress_bar:

            total_count = dataframe.shape[0]
            train_count = math.floor(total_count * (1-test_size))
            test_count = total_count - train_count

            train_dataframe = dataframe.head(train_count)
            test_dataframe = dataframe.tail(test_count)

            train_slices = slice_dataframe(name, train_dataframe, slice_width, window_step)
            test_slices = slice_dataframe(name, test_dataframe, slice_width, window_step)

            train_dataframes.update(train_slices)
            test_dataframes.update(test_slices)

            train_dataframes_dict[name] = train_slices
            test_dataframes_dict[name] = test_slices

        progress_bar.close()

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(f"{class_name}.{function_name} splitted train:{str(len(train_dataframes))}, "
                            f"test: {str(len(test_dataframes))}")

        return train_dataframes, test_dataframes
