from datetime import datetime

class TrackingDecorator(object):

    def track_time(func):

        def wrap(self, *args, **kwargs):
            start_time = datetime.now()
            print(func.__qualname__ + " started")

            result = func(self, *args, **kwargs)

            time_elapsed = datetime.now() - start_time
            print(func.__qualname__ + " finished in {}".format(time_elapsed))

            return result

        return wrap
