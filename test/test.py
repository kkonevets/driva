import pickle
import xml.etree.ElementTree as ET
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
import redis
import requests
import multiprocessing as mp
from functools import partial

import ai.features as ft


def tripple_natural_keys(tripple):
    """
    :type text: str
    """
    return [ft.natural_keys(tripple[0])]


def zipped_tripples(data_dir, n1='RichTrack', n2='RichTrackPoints',
                    n3='IncomingTrackPoints'):
    tracks_files = sorted(glob('%s/*%s.csv*' % (data_dir, n1)))
    richpoints_files = sorted(glob('%s/*%s.csv*' % (data_dir, n2)))
    points_files = sorted(glob('%s/*%s.csv*' % (data_dir, n3)))
    zipped = zip(tracks_files, richpoints_files, points_files)
    zipped_files = [(a, b, c) for a, b, c in zipped]

    zipped_files.sort(key=tripple_natural_keys)

    return zipped_files


def prepare_db(save_dir, r):
    zipped_files = zipped_tripples(save_dir)
    tracks_files = sorted(glob('%s/*_%s.csv*' % (save_dir, 'RichTrack')))

    # set inner test tokens
    all_info = pd.concat([pd.read_csv(f) for f in tracks_files])
    tokens = all_info['DeviceToken'].unique()
    r.delete('inner_test_tokens')
    out = []
    for t in tokens:
        out.append(r.lpush('inner_test_tokens', t.lower()))

    model = pickle.load(open("./data/models/model_busses.pickle", "rb"))
    r.set("model_bus_detection", pickle.dumps(model))

    return zipped_files


def post_files(files_list):
    res = []
    for fnames in files_list:
        di = post_files_worker(fnames)
        res.append(di)
    return pd.concat(res)


def post_files_worker(fnames):
    a, b, c = fnames
    rich = pd.read_csv(a)
    Id, token = rich[['Id', 'DeviceToken']].iloc[0]
    files = {'RichTracks': open(a, 'rb'),
             'RichTrackPoints': open(b, 'rb'),
             'IncomingTrackPoints': open(c, 'rb')}

    req = requests.post("http://0.0.0.0:1180", files=files)
    root = ET.fromstring(req.text)
    pred = root.find("prediction").text
    error = root.find("error").text

    pred = error if pred is None else pred

    print(pred)
    return pd.DataFrame([[Id, token,  pred]], columns=['RichTrackId', 'token', 'pred'])


def split_frames_by_track(path_to_tracks, path_to_rich, path_to_incoming, save_dir):
    track_info = pd.read_csv(path_to_tracks)
    rich = pd.read_csv(path_to_rich)
    incoming = pd.read_csv(path_to_incoming)

    # track_info = track_info[track_info['TrackOrigin'] == 'Passanger']

    ids = track_info[['Id', 'IncomingTrackId']].values
    for rich_id, inc_id in ids:
        _info = track_info[track_info['Id'] == rich_id]
        _rich = rich[rich['RichTrackId'] == rich_id]
        _incoming = incoming[incoming['IncomingTrackId'] == inc_id]

        _info.to_csv(save_dir + '/%s_RichTrack.csv.gz' % rich_id,
                     compression='gzip', index=False, encoding='utf8')
        _rich.to_csv(save_dir + '/%s_RichTrackPoints.csv.gz' % rich_id,
                     compression='gzip', index=False, encoding='utf8')
        _incoming.to_csv(save_dir + '/%s_IncomingTrackPoints.csv.gz' % rich_id,
                         compression='gzip', index=False, encoding='utf8')


def get_features():
    import streem.processing as proc
    path_to_tracks = '/home/guyos/Downloads/MashaTrack/RichTrackMasha2.csv'
    path_to_rich = '/home/guyos/Downloads/MashaTrack/RichTrackPointsMasha2.csv'
    path_to_incoming = '/home/guyos/Downloads/MashaTrack/IncomingMashaTrain2.csv'
    _, _, _, ftrs = proc.get_driver_features_from_files(path_to_tracks, path_to_rich,
                                                        path_to_incoming)


if __name__ == '__main__':
    data_dir = '/home/guyos/Yandex.Disk/raxel/data/'
    save_dir = '/home/guyos/Downloads/save/'
    r = redis.Redis(host='localhost')

    zipped_users = zipped_tripples(data_dir, n1='tracks', n2='rich', n3='incoming')
    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        print(path_to_tracks)
        split_frames_by_track(path_to_tracks, path_to_rich, path_to_incoming, save_dir)

    zipped_files = prepare_db(save_dir, r)

    cpu_count = mp.cpu_count() + 1
    pool = mp.Pool()
    func = partial(post_files)
    result = pool.map(func, np.array_split(zipped_files, cpu_count))
    pool.close()
    pool.join()

    res_df = pd.concat(result)
    res_df.sort_values(['token', 'RichTrackId'], inplace=True)
    res_df.to_csv(data_dir + 'predictions.csv', index=False)

    Counter(res_df['pred'])
    # len(res_df)

    # cop = res_df.copy()
    # Counter(cop[1])

    # np.all(cop == res_df)

    # set(res_df[res_df[1] == 'Train'][0]).intersection(cop[cop[1] == 'Train'][0])



    l = []
    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        l.append(pd.read_csv(path_to_tracks))

    df = pd.concat(l)

    busses = pd.read_csv('/home/guyos/Documents/bus_detection/50_50/data_tracks.csv.gz')

    lii = list(set(df['Id']).intersection(busses['Id']))
    # set(busses['IncomingTrackId']).symmetric_difference(X.index)

    pp = pd.read_csv(data_dir + 'predictions1.csv')
    pp = pp[(pp['RichTrackId'].isin(lii)) & (pp['pred']=='OriginalDriver')]
    Counter(pp['pred'])