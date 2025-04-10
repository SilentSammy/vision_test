import time

class Stopwatch:
    def __init__(self):
        self._start_time = None

    def start(self):
        self._start_time = time.time()
        # print("Stopwatch started.")

    def stop(self):
        if self._start_time is None:
            print("Stopwatch hasn't been started!")
        else:
            elapsed = time.time() - self._start_time
            print(f"Stopwatch stopped. Elapsed time: {elapsed:.2f} seconds.")
            self._start_time = None