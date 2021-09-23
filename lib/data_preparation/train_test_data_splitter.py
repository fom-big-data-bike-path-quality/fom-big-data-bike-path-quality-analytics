import inspect

from sklearn.model_selection import train_test_split
from tracking_decorator import TrackingDecorator


#
# Main
#

class TrainTestDataSplitter:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, test_size=0.15, validation_size=0.10, random_state=0, quiet=False):
        ids = sorted(list(dataframes.keys()))

        train_and_validation_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
        train_ids, validation_ids = train_test_split(train_and_validation_ids, test_size=validation_size, random_state=random_state)

        train_dataframes = {id: dataframes[id] for id in train_ids}
        validation_dataframes = {id: dataframes[id] for id in validation_ids}
        test_dataframes = {id: dataframes[id] for id in test_ids}

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " splitted "
                            + "train: " + str(len(train_dataframes)) + ", "
                            + "validation:" + str(len(validation_dataframes)) + ", "
                            + "test:" + str(len(test_dataframes)))

        return train_dataframes, validation_dataframes, test_dataframes
