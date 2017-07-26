#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 04:00:50 2016

@author: guyos
"""

import os
import pickle

import ai.cmatching as cmatching
import numpy as np
import pandas as pd
import redis
from geopandas import GeoDataFrame
from haversine import haversine
from scipy.stats import kurtosis, skew
from shapely.geometry import Point

import ai.features as ft
import ai.qpostgis as qpg
import ai.ways as ways
import ai.tripmatching as tm
from ai.company_rdp import rdp

r = redis.Redis(host='localhost')


def annotate_point(ax, text, point):
    ax.annotate(text, xy=point, xytext=(20, -20), textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3,rad=0.5', 'color': 'red'})


def plot_bus_stops(track, bus_stops, close_all=True):
    from matplotlib import pyplot as plt
    from matplotlib import cm

    if close_all:
        plt.close("all")
    f, ax = plt.subplots()
    f.set_size_inches(18.67, 9.86)
    color = [str(item) for item in track.Speed]
    sc = ax.scatter(track.Longitude, track.Latitude, s=20, c=color, cmap=cm.rainbow)
    plt.colorbar(sc, orientation='vertical')
    ax.set_title(str(track.IncomingTrackId.unique()[0]))
    if len(bus_stops) > 0:
        ax.plot(bus_stops[:, 0], bus_stops[:, 1], '*', markersize=15, color='grey', alpha=0.4)
    annotate_point(ax, 'Start', track.iloc[0][['Longitude', 'Latitude']])
    annotate_point(ax, 'End', track.iloc[-1][['Longitude', 'Latitude']])
    f.tight_layout()

    return f


def project_stops(track, stops):
    coords = track[['Longitude', 'Latitude']]
    coords = coords.drop_duplicates()

    ind = [np.argmin(np.linalg.norm(coords - pt, axis=1)) for pt in stops]
    return ind


def get_stop_ftrs(track, proj_idxs):
    ftrs = {}

    locality = np.array([(i - 7, i + 3) for i in proj_idxs])

    max_idx = len(track)

    # for consistency
    locality[locality < 0] = 0
    locality[locality > max_idx] = max_idx

    l = len(locality)
    if l:
        speeds = [min(track.iloc[left:right + 1].Speed) for
                  left, right in locality]
    else:
        speeds = []

    ftrs['bus_stop_speed_mean'] = np.mean(speeds) if l else -999
    ftrs['bus_stop_speed_median'] = np.median(speeds) if l else -999
    ftrs['bus_stop_speed_std'] = np.std(speeds) if l else -999
    ftrs['bus_stop_speed_kurtosis'] = kurtosis(speeds) if l else -999
    ftrs['bus_stop_speed_skew'] = skew(speeds) if l else -999
    ftrs['bus_stop_num'] = l

    return ftrs


def get_data_dir():
    if '__file__' in globals():
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.dirname(path)
    else:
        path = '.'
    path = os.path.join(path, 'data')

    return path


def get_country_by_lat_lon(lat, lon):
    borders_blob = r.get('world_borders')
    if borders_blob is None:
        fname = os.path.join(get_data_dir(),
                             "TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp")
        borders = GeoDataFrame.from_file(fname)

        r.set('world_borders', pickle.dumps(borders))
    else:
        borders = pickle.loads(borders_blob)

    point = Point((lon, lat))
    index = borders['geometry'].intersects(point)
    names = borders[index]['ISO2'].values

    return names


def get_left_traffic_countries():
    countries_blob = r.get('left_traffic_countries')
    if countries_blob is None:
        fname = os.path.join(get_data_dir(),
                             'left_traffic_counties.csv')
        countries = pd.read_csv(fname)
        r.set('left_traffic_countries', pickle.dumps(countries))
    else:
        countries = pickle.loads(countries_blob)

    return countries['ISO2']


def is_left_traffic_country(lat, lon):
    cnames = get_country_by_lat_lon(lat, lon)
    left_countries = get_left_traffic_countries()

    return len(set(cnames).intersection(left_countries)) > 0


def get_bus_stops(track, host='localhost'):
    # determin country driving side

    # if np.median(track.Longitude) < 20:  # Singapur
    #     left_side = True
    #     right_side = False
    # else:  # Russia
    #     left_side = False
    #     right_side = True

    lat, lon = np.median(track.Latitude), np.median(track.Longitude)
    left_side = is_left_traffic_country(lat, lon)
    right_side = not left_side

    # reduse points number
    track_rdp = np.array(rdp(track[['Longitude', 'Latitude']],
                             epsilon=10 * tm.meter))

    # compute polygons
    poly = cmatching.get_convex_hull(track_rdp, threshold=50 * tm.meter,
                                     left_side=left_side,
                                     right_side=right_side)

    bus_stops, names = qpg.get_stops_in_geom(poly, host=host)

    return bus_stops, names


def get_geoftrs(track: pd.DataFrame, ped_start=None,
                host: str = 'localhost') -> pd.DataFrame:
    ftrs = {}

    bus_stops, names = get_bus_stops(track, host)

    # get bus stops in radius at the beginning of pedestrian track
    if ped_start is not None:
        # got_stop - number of stops in radius 50 meters from pedestrian start point
        got_stop = len([1 for stop in bus_stops if
                        1000 * haversine((stop[1], stop[0]), ped_start) < 50])
    else:
        got_stop = -1
    ftrs['ped_last_stop'] = got_stop

    proj_idxs = project_stops(track, bus_stops)

    stop_ftrs = get_stop_ftrs(track, proj_idxs)
    way_ftrs = ways.way_features(track, host=host)

    ftrs.update(stop_ftrs)
    ftrs.update(way_ftrs)

    ftrs_df = ft.dict_to_frame(ftrs)
    return ftrs_df


if __name__ == "__main__":
    host = 'localhost'
    #
    # points = pd.read_csv('/home/guyos/Documents/bus_detection/busses/IncomingTrackPointsBus.zip')
    # points['StartDate'] = 0
    # points.drop_duplicates(['IncomingTrackId', 'PointDate'], inplace=True)
    # points.sort_values(['IncomingTrackId', 'PointDate'], inplace=True)
    #
    # grouped = points.groupby(['IncomingTrackId'])
    # track = grouped.get_group(5492)
    #
    # wrapper = partial(get_geoftrs, host=host)
    # geo_ftrs = grouped.apply(wrapper)
    # geo_ftrs.index = geo_ftrs.index.droplevel(1)
    #
    # lat, lon = -7.1008338, 107.4703494
    # is_left_traffic_country(lat, lon)

    # print(get_data_dir())

    # track = pd.read_csv('/home/guyos/Yandex.Disk/company/data/track.csv')
    # track = ways.delete_outliers_from_track(track)
