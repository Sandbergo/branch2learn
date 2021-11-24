"""
General utilities.

by Lars Sandberg @Sandbergo
May 2021
"""

import datetime


class Logger:
    def __init__(self, filename=None):
        self.logfile = f"{filename}_{self.format_time(file=True)}.txt"

    def format_time(self, file=False) -> str:
        time = datetime.datetime.now()
        if file:
            time_string = time.strftime("%m%d_%H%M%S")
        else:
            time_string = time.strftime("%m-%d %H:%M:%S.%f")
            time_string = time_string[:-4]
        return time_string

    def __call__(self, string: str) -> None:
        string = f"[{self.format_time()}] {string}"
        print(string)
        if self.logfile is not None:
            with open(self.logfile, mode="a") as out_file:
                print(string, file=out_file)
