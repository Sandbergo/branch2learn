import datetime


class Logger:
    def __init__(self, filename=None):
        self.logfile = f'{filename}_{self.format_time(file=True)}.txt'

    def format_time(self, file=False):
        t = datetime.datetime.now()
        if file:
            s = t.strftime('%m%d_%H%M%S')
        else:
            s = t.strftime('%m-%d %H:%M:%S.%f')
            s = s[:-4]
        return s

    def __call__(self, str: str):
        str = f'[{self.format_time()}] {str}'
        print(str)
        if self.logfile is not None:
            with open(self.logfile, mode='a') as f:
                print(str, file=f)
