import locale
import re
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd


@contextmanager
def setlocale():
    saved = locale.setlocale(locale.LC_ALL)
    try:
        yield locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    finally:
        locale.setlocale(locale.LC_ALL, saved)


def extract(line):
    dates, mscs = re.findall(date_exp, line), re.findall(time_exp, line)
    if len(dates) != 1 or len(mscs) != 1:
        return None, None
    date_str = dates[0]
    date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")
    t = float(mscs[0])

    return date, t


if __name__ == '__main__':
    file_path = '/var/log/uwsgi/driva.log'
    # file_path = '/home/guyos/Downloads/driva.log'

    date_exp = 'bytes\} \[(.*?)\] POST / => generated'
    time_exp = ' bytes in (.*?) msecs \(HTTP/'

    with setlocale() and open(file_path, 'r') as f:
        parsed = [extract(line) for line in f]

    df = pd.DataFrame(parsed, columns=['date', 'mscs'])
    df = df[np.sum(df.isnull(), axis=1) == 0]

    with setlocale():
        point = datetime.strptime('Thu Apr 27 19:24:34 2017', "%a %b %d %H:%M:%S %Y")
    old = df[df.date < point]
    new = df[df.date > point]

    print('Before %s, mean %s, std %s, median %s, max %s' % (
        point, old.mscs.mean(), old.mscs.std(), old.mscs.median(), old.mscs.max()))
    print('After  %s, mean %s, std %s, median %s, max %s' % (
        point, new.mscs.mean(), new.mscs.std(), new.mscs.median(), new.mscs.max()))
