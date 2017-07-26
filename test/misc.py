import pandas as pd
import numpy as np
import redis
import datetime

r = redis.StrictRedis(host='localhost')
keys = r.keys(pattern='*info*')

l = []
for key in keys:
    d = r.hgetall(key)
    if b'prediction'in d.keys() and b'StartDate' in d.keys() and d[b'prediction'] not in (b'OriginalDriver'):
        bdate = d[b'StartDate']
        num = np.int64(float(bdate.decode("utf-8")))
        sdate = datetime.datetime.fromtimestamp(num).strftime('%Y-%m-%d %H:%M:%S')
        l.append((d[b'IncomingTrackId'].decode("utf-8"), sdate, d[b'prediction'].decode("utf-8"), num))

df = pd.DataFrame(l, columns=['IncomingTrackId', 'StartDate', 'prediction', 'timestamp'])
df.sort_values(by='timestamp', axis=0, inplace=True)
df.drop('timestamp', inplace=True, axis=1)
df.to_csv('/home/raxel/preds.csv', index=False)
