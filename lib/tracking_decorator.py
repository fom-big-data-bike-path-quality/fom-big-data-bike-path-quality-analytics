from datetime import datetime


class TrackingDecorator(object):

    def track_time(func):

        def wrap(self, *args, **kwargs):
            if "logger" in kwargs:
                logger = kwargs["logger"]
            else:
                logger = None

            start_time = datetime.now()

            if logger is not None:
                logger.log_line(f"\n{func.__qualname__} started")
            else:
                print(f"\n{func.__qualname__} started")

            result = func(self, *args, **kwargs)

            time_elapsed = datetime.now() - start_time

            if logger is not None:
                logger.log_line(f"{func.__qualname__} finished in {time_elapsed}")
            else:
                print(f"{func.__qualname__} finished in {time_elapsed}")

            return result

        return wrap
