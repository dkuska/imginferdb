import time
import functools
import memory_profiler
from utils.performance_monitor import log_performance_metrics


class ProfilerWrapper:
    def __init__(self, model):
        self.model = model

    def profile_function(self, func):
        """
        Decorator to measure the execution time and memory usage of a function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            mem_usage_before = memory_profiler.memory_usage()[0]

            result = func(*args, **kwargs)

            mem_usage_after = memory_profiler.memory_usage()[0]
            end_time = time.perf_counter()

            duration = end_time - start_time
            mem_usage = mem_usage_after - mem_usage_before

            # Logging the measured performance metrics
            log_performance_metrics(func.__name__, duration, mem_usage)

            return result

        return wrapper

    def __getattr__(self, name):
        """
        If the method is part of the profiling list, return a wrapped version.
        Otherwise, return the method as is.
        """
        attr = getattr(self.model, name)

        if callable(attr):
            return self.profile_function(attr)
        return attr
