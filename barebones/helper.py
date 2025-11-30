import time
from typing import Tuple
from functools import wraps

def measure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        delta = time.time() - start
        hours, minutes, seconds = seconds_to_time(delta)
        print(f"\nElapsed time: {delta:.2f}s => {hours:02}:{minutes:02}:{seconds:02}")
        return result
    return wrapper

def seconds_to_time(seconds: float) -> Tuple[int, int, int]:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return int(hours), int(minutes), int(seconds)
