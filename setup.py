import os
import pickle

import pandas as pd
import redis

if __name__ == '__main__':
    # generate cython binaries
    os.chmod("./ai/csetup.py", 777)
    os.system('python ./ai/csetup.py build_ext -b ./ai clean --all')

    # connect Redis
    r = redis.Redis(host='localhost')

    # set inner test tokens
    tokens = pd.read_csv('./test/test_tokens.csv')
    r.delete('inner_test_tokens')
    for t in tokens.DeviceToken:
        r.lpush('inner_test_tokens', t)

    # load model
    model = pickle.load(open("./data/models/model_busses.pickle", "rb"))
    r.set('model_bus_detection', pickle.dumps(model))
