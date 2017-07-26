#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:25:57 2016

REDIS OPERATIONS

@author: guyos
"""

import pandas as pd
import redis

r = redis.Redis(host='localhost')


def get_list_data(token, tag, start=0, end=-1):
    lpoints = r.lrange('%s_%s' % (token, tag), start, end)
    if len(lpoints) == 0:
        return None
    points = pd.concat([pd.read_msgpack(el) for el in lpoints])
    if 'PointDate' in points.columns:
        points.PointDate = pd.to_datetime(points.PointDate)
    return points


def get_last_data(device_token, tag):
    # get value of highest score
    ids = r.zrevrange('%s.IncomingTrackId.EndDate.index' % device_token, 0, 0)

    if len(ids) == 0:
        return None

    incoming_ids = [b.decode("utf-8") for b in ids]

    df = mget_tag(device_token, incoming_ids, tag)
    return df


def delete_from_list(name, indexes):
    for idx in indexes:
        r.lset(name, idx, 0)
        r.lrem(name, 0, 0)


def list_keys(device_token, incoming_ids, tag):
    tag_keys = ['%s:%s:%s' % (device_token, inc_id,
                              tag) for inc_id in incoming_ids]
    return tag_keys


def mget_tag(device_token, incoming_ids, tag, parser=pd.read_msgpack):
    def get_collection_by_parser(_parser):
        if _parser == pd.read_msgpack:
            _collection = pd.DataFrame()
        else:
            _collection = pd.Series()
        return _collection

    keys = list_keys(device_token, incoming_ids, tag)
    if len(keys) > 0:
        blobs = r.mget(keys)
        blobs = list(filter(lambda b: b is not None, blobs))
        if len(blobs):
            collection = pd.concat([parser(b) for b in blobs])
        else:
            collection = get_collection_by_parser(parser)
    else:
        collection = get_collection_by_parser(parser)

    return collection
