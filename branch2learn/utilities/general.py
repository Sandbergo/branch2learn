import datetime
import numpy as np


class Logger:
    def __init__(self, filename=None):
        self.logfile = f'{filename}_{self.format_time(file=True)}.txt'

    def format_time(self, file=False) -> str:
        t = datetime.datetime.now()
        if file:
            s = t.strftime('%m%d_%H%M%S')
        else:
            s = t.strftime('%m-%d %H:%M:%S.%f')
            s = s[:-4]
        return s

    def __call__(self, str: str) -> None:
        str = f'[{self.format_time()}] {str}'
        print(str)
        if self.logfile is not None:
            with open(self.logfile, mode='a') as f:
                print(str, file=f)


def shifted_geometric_mean(iterable, alpha):
    """
    Calculates shifted geometric mean.
    :param iterable:        ordered collection of values which will be used for SGM calculation
    :param alpha:           value added to each element included in calculation of geometric mean
    :return                 value of shifted geometric mean
    """

    a = np.add(iterable, alpha);
    a = np.log(a)
    return np.exp(a.sum() / len(a)) - alpha