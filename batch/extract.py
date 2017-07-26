#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 01:50:03 2016

@author: Kirill Konevets
"""

import pickle
from functools import partial

import pandas as pd
import redis

import ai.features as ft
import ai.geoftrs as gf
import ai.tripmatching as tm
import streem.processing as proc
import streem.qredis as qredis


def batch_extract_features(path_to_tracks, path_to_rich, path_to_incoming,
                           use_geo_ftrs=False, host='localhost'):
    track_info, rich_points, incoming_points = \
        proc.load_data(path_to_tracks, path_to_rich, path_to_incoming)

    incoming_points.rename(columns={'IncomingTrackId': 'TrackId'}, inplace=True)
    rich_points.rename(columns={"RichTrackId": "TrackId"}, inplace=True)

    if 'Id' in rich_points.columns:
        del rich_points['Id']
    rich_points = rich_points.merge(track_info[['StartDate', 'Id']],
                                    left_on='TrackId', right_on='Id')
    del rich_points['Id']
    # smoothed = tm.apply_rdp(rich_points, epsilon=50)
    # smoothed.sort_values('StartDate', inplace=True)

    # select columns
    usecols = ['Id', 'IncomingTrackId', 'StartDate', 'Distance',
               'Duration', 'AccelerationCount', 'DecelerationCount', 'Rating', 'PhoneUsage',
               'DistanceGPS', 'Urban', 'OverSpeedMileage',
               'MidOverSpeedMileage', 'HighOverSpeedMileage']
    track_info = track_info[usecols]

    # main: get track features
    ftrs = ft.get_driver_features((incoming_points, rich_points, track_info),
                                  use_geo_ftrs=use_geo_ftrs, host=host)

    return ftrs


# %%

def main():
    r = redis.Redis(host='localhost')

    incoming_keys = r.keys('*incoming*')
    tokens = [el.decode("utf-8").split(':')[0] for el in incoming_keys]

    if len(incoming_keys) == 0:
        return

    for DeviceToken in tokens:
        print(DeviceToken)

        index_key = '%s.IncomingTrackId.EndDate.index' % DeviceToken
        incoming_ids = [int(i.decode('utf8')) for i in
                        r.zrange(index_key, 0, -1)]

        incoming_points = qredis.mget_tag(DeviceToken, incoming_ids, 'incoming')
        rich_points = qredis.mget_tag(DeviceToken, incoming_ids, 'rich')
        track_info = qredis.mget_tag(DeviceToken, incoming_ids, 'track')

        incoming_points.rename(columns={'IncomingTrackId': 'TrackId'}, inplace=True)
        rich_points.rename(columns={"RichTrackId": "TrackId"}, inplace=True)

        # SORT BY DATE BEFORE SMOOTHING
        if 'Id' in rich_points.columns:
            del rich_points['Id']
        rich_points = rich_points.merge(track_info[['StartDate', 'Id']],
                                        left_on='TrackId', right_on='Id')
        del rich_points['Id']
        smoothed = tm.apply_rdp(rich_points, epsilon=50)
        smoothed.sort_values('StartDate', inplace=True)
        polygons = tm.get_polygons(smoothed, threshold=30)

        grouped = track_info.groupby('IncomingTrackId')
        for tup in grouped:
            IncomingTrackId, info = tup
            TrackId = info.iloc[0].Id
            base_name = '%s:%s' % (DeviceToken, IncomingTrackId)
            sm = smoothed[smoothed.TrackId == TrackId]
            pl = polygons[TrackId]
            r.set('%s:%s' % (base_name, 'smoothed'),
                  sm.to_msgpack(compress='zlib', encoding='utf8'))
            r.set('%s:%s' % (base_name, 'polygon'), pickle.dumps(pl))

        # select columns
        usecols = ['Id', 'IncomingTrackId', 'StartDate', 'Distance',
                   'Duration', 'AccelerationCount', 'DecelerationCount', 'Rating', 'PhoneUsage',
                   'DistanceGPS', 'Urban', 'OverSpeedMileage',
                   'MidOverSpeedMileage', 'HighOverSpeedMileage']
        track_info = track_info[usecols]

        # main: get track features
        ftrs = ft.get_driver_features((incoming_points, rich_points, track_info), True)

        # save features to database
        grouped = ftrs.groupby(ftrs.index)
        for tup in grouped:
            IncomingTrackId, ftr = tup
            ftrc = ftr.copy()
            ftrc.insert(0, 'IncomingTrackId', IncomingTrackId)
            key = '%s:%s:%s' % (DeviceToken, IncomingTrackId, 'features')
            blob = ftrc.to_msgpack(compress='zlib', encoding='utf8')
            r.set(key, blob)


# %%

if __name__ == "__main__":
    main()
