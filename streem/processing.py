#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:31:21 2016

Online data processing, working with data base

@author: guyos
"""

import os
import pickle

import pandas as pd
import redis
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy import create_engine
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER as GUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import ai.features as ft
import ai.taxi_airport as taxi_airport
import ai.bicycle as bicycle

Base = declarative_base()

SERVER = ''
USER = ''
MSSQLPASSWORD = ''


# %% FILE SYSTEM FUNCTIONS

def read_points(path_to_points, usecols=None):
    points = pd.read_csv(path_to_points, usecols=usecols, encoding='utf8',
                         na_values=['', 'NA'])
    if 'PointDate' in points.columns:
        points.PointDate = pd.to_datetime(points.PointDate)
    return points


def load_data(path_to_tracks, path_to_rich, path_to_incoming):
    incoming_points = read_points(path_to_incoming)
    rich_points = read_points(path_to_rich)
    track_info = read_points(path_to_tracks)
    return track_info, rich_points, incoming_points


# %% CLASSES


class RichTrack(Base):
    __tablename__ = 'RichTracks'

    id = Column(Integer, primary_key=True)
    Prediction = Column(String)

    def __repr__(self):
        return "<track(id='%s', prediction='%s')>" % (
            self.id, self.Prediction)


class RichTrackFeature(Base):
    __tablename__ = 'RichTrackFeatures'

    id = Column(Integer, primary_key=True)
    DeviceToken = Column(GUID)
    RichTrackId = Column(Integer)
    Key = Column(String)
    Value = Column(Float)

    def __repr__(self):
        return "<feature(DeviceToken='%s', RichTrackId='%s', Key='%s', Value='%s')>" % \
               (self.DeviceToken, self.RichTrackId, self.Key, self.Value)


class Features(object):
    _mssql_engine = None
    _mssql_connection = None
    _redis_connection = None

    threshold = 0.4
    rail_frequency = 0.8

    def __init__(self, path_to_tracks, path_to_rich,
                 path_to_incoming, host='localhost'):
        self.path_to_tracks = path_to_tracks
        self.path_to_rich = path_to_rich
        self.path_to_incoming = path_to_incoming
        self.host = host
        self.error = ''
        self.DeviceToken = None
        self.IncomingTrackId = None
        self.ftrs = []
        self.pred = None
        self.debug_pred = None
        self.company_id = None

        self._connect()

        self.track_info, self.rich_points, self.incoming_points = \
            load_data(path_to_tracks, path_to_rich, path_to_incoming)

        # rename for compatibility with old code
        self.incoming_points['TrackId'] = self.incoming_points['IncomingTrackId']
        self.rich_points['TrackId'] = self.rich_points['RichTrackId']

        tokens = self.track_info['DeviceToken'].unique()
        if len(tokens) != 1:
            self.error = 'got several DeviceTokens %s' % len(tokens)
            return
        self.DeviceToken = tokens[0].lower()

        ids = self.track_info['Id'].unique()
        if len(ids) != 1:
            self.error = 'got number of %s RichTrackIds, expected 1' % len(ids)
            return

        self.RichTrackId = int(ids[0])
        self.company_id = self.track_info['CompanyId'].iloc[0]
        self.IncomingTrackId = self.track_info['IncomingTrackId'].iloc[0]

        # main: get track features
        self.extract()

    def _connect(self):
        self._redis_connection = redis.Redis(host='localhost')

        con_string = "mssql+pyodbc://%s:%s@company-ab-sql/?trusted_connection=no"
        self._mssql_engine = create_engine(con_string % (USER, MSSQLPASSWORD))
        self._mssql_connection = self._mssql_engine.connect()

    def close_connections(self):
        if self._mssql_connection is not None:
            self._mssql_connection.close()
        if self._mssql_engine is not None:
            self._mssql_engine.dispose()
        self._mssql_engine = None
        self._mssql_connection = None
        self._redis_connection = None

    def extract(self):
        if self.error:
            print(self.error)
            return
        frames = self.incoming_points, self.rich_points, self.track_info
        self.ftrs = ft.get_driver_features(frames, use_geo_ftrs=True, host=self.host)

        is_taxi = taxi_airport.is_taxi_airport(self.rich_points, self._mssql_connection)
        if is_taxi == 1:
            self.debug_pred = 'Taxi'
        elif is_taxi == -1:
            self.debug_pred = 'Other'

        is_bicycle = bicycle.is_bicycle(self.incoming_points)
        if is_bicycle:
            self.debug_pred = 'Bicycle'

    def predict(self):
        if self.debug_pred is None:
            self.debug_pred = self.predict_common()
        elif len(self.ftrs) == 0:
            self.debug_pred = None
            self.error = 'track is empty'

        # get test tokens
        blobs = self._redis_connection.lrange('inner_test_tokens', 0, -1)
        inner_test_tokens = [b.decode("utf-8") for b in blobs]
        device_token = self.DeviceToken.lower()
        if device_token in inner_test_tokens:
            self.pred = self.debug_pred
        elif self.company_id in (6, 1064) and self.debug_pred != 'Bicycle':  # VSK = 6
            self.pred = self.debug_pred
        else:
            self.pred = 'OriginalDriver'

    def predict_common(self):
        _pred = 'OriginalDriver'

        if len(self.ftrs) == 0:
            return _pred

        track_ftrs = self.ftrs.iloc[0]

        if track_ftrs.speed_max < 25:  # km/h
            _pred = 'Other'

        # if 0.2 > track_ftrs.rel_end_count_100000 > 0:
        #     _pred = 'Other'

        if _pred == 'OriginalDriver':
            blob = self._redis_connection.get('model_bus_detection')
            assert blob is not None
            columns, model_bus = pickle.loads(blob)
            sub_ftrs = self.ftrs[columns]
            proba = model_bus.predict_proba(sub_ftrs)[0, 1]
            y = int(proba >= self.threshold)
            if y == 1 and track_ftrs['DistanceGPS'] > 2:
                _pred = 'Bus'
            if 'rail_frequency' in track_ftrs:
                # TODO: train model with rail_frequency, don't use if-else
                if track_ftrs['rail_frequency'] >= self.rail_frequency and \
                                track_ftrs['DistanceGPS'] > 2:  # more than 80%
                    _pred = 'Train'

        return _pred

    def __del__(self):
        os.remove(self.path_to_tracks)
        os.remove(self.path_to_rich)
        os.remove(self.path_to_incoming)

        self.close_connections()


def save_features(features):
    """
    Save features and debug prediction to data base
    :param features: Features class instance
    """
    if len(features.ftrs) == 0:
        return

    con_string = "mssql+pyodbc://%s:%s@company-ab-sql/?trusted_connection=no"
    engine = create_engine(con_string % (USER, MSSQLPASSWORD))
    Session = sessionmaker(bind=engine)
    session = Session()

    # # delete old features if present
    # num_deleted = session.query(RichTrackFeature).\
    #     filter_by(RichTrackId=features.RichTrackId).\
    #     delete(synchronize_session='evaluate')
    # print('num_deleted %s, RichTrackId %s' % (num_deleted, features.RichTrackId))
    # if num_deleted:
    #     print('deleting % old features for %s rtid' %
    #           (num_deleted, features.RichTrackId))

    # save features to database
    X = features.ftrs.copy()
    X.index = ['value']
    for key, row in X.T.iterrows():
        feature = RichTrackFeature(DeviceToken=features.DeviceToken,
                                   RichTrackId=features.RichTrackId,
                                   Key=key,
                                   Value=row.value)
        session.add(feature)

    if features.debug_pred is not None:
        rich_track = session.query(RichTrack). \
            filter_by(id=features.RichTrackId).first()
        if rich_track:
            rich_track.Prediction = features.debug_pred

    session.commit()

    session.close()
    engine.dispose()

# %%

# path_to_tracks = '/home/guyos/Downloads/save/2666_RichTrack.csv.gz'
# path_to_rich = '/home/guyos/Downloads/save/2666_RichTrackPoints.csv.gz'
# path_to_incoming = '/home/guyos/Downloads/save/2666_IncomingTrackPoints.csv.gz'
# _,_,_,ftrs = get_driver_features_from_files(path_to_tracks, path_to_rich,
#                                       path_to_incoming, host='localhost')


# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column, Integer, String
# from sqlalchemy.orm import sessionmaker
# import pandas as pd
#
# SERVER = ''
# USER = ''
# MSSQLPASSWORD = ''
#
# engine = create_engine("mssql+pyodbc://%s:%s@company-ab-sql/?trusted_connection=no" % (USER, MSSQLPASSWORD))
#
# Session = sessionmaker(bind=engine)
# session = Session()
#
#
# our_user = session.query(RichTrack).filter_by(id=54).first()
# our_user.Prediction = 'test'
# session.commit()


# # conn = engine.connect()

# df = pd.DataFrame([['9fbbbd45-1e8d-4430-92b5-96f8fd119b7c', '23455', 'fdff', 33]],
#                   columns=['DeviceToken', 'RichTrackId', 'Key', 'Value'])
#
# df.to_sql('RichTrackFeatures', con=engine, index=False, if_exists='append')
#
# # with create_engine('mssql+pyodbc://%s:%s@%s' % (USER, MSSQLPASSWORD, SERVER)) as engine:
# #     print(engine)


# ПЕРЕНЕСТИ PREDICTION в PROCESSING class method
