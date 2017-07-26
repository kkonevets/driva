# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:12:33 2016

@author: Kirill Konevets
email:   kkonevets@gmail.com

CONTENT: FEATURE EXTRACTION
    
"""

import re
from functools import partial

import numpy as np
import pandas as pd
from haversine import haversine
from sklearn.metrics.pairwise import euclidean_distances

import ai.tripmatching as tm
import ai.ways as ways

# ==============================================================================
# main
# ==============================================================================


def get_driver_features(frames, use_geo_ftrs=False, host='localhost'):
    incoming_points, rich_points, track_info = frames

    # get features from incoming points
    grouped = incoming_points.groupby('TrackId')
    wrapper = partial(get_incoming_features,
                      use_geo_ftrs=use_geo_ftrs, host=host)
    incoming_ftrs = grouped.apply(wrapper)
    incoming_ftrs.reset_index(level=1, drop=True, inplace=True)
    incoming_ftrs.fillna(0, inplace=True)

    # get features from rich points
    grouped = rich_points.groupby('TrackId')
    rich_ftrs = grouped.apply(get_rich_features)
    rich_ftrs.reset_index(level=1, drop=True, inplace=True)

    # dists = get_distance_from_last_track(rich_points, last_track=last_track)
    # rich_ftrs = rich_ftrs.merge(dists, left_index=True, right_index=True)

    # end_counts = get_end_counts(smoothed)
    # ftrs = rich_ftrs.merge(end_counts, left_index=True, right_index=True)

    # add track info
    info = extract_track_info(track_info)
    ftrs = rich_ftrs.merge(info, left_index=True, right_on=['TrackId'])
    del ftrs['TrackId']
    ftrs.rename(columns={"IncomingTrackId": "TrackId"}, inplace=True)
    ftrs = incoming_ftrs.merge(ftrs, left_index=True, right_on=['TrackId'])
    ftrs.set_index('TrackId', inplace=True)

    colnames = list(ftrs.columns)
    colnames.sort(key=natural_keys)
    ftrs = ftrs[colnames]

    return ftrs


# ==============================================================================
# incoming features
# ==============================================================================

def get_incoming_features(track, use_geo_ftrs=False, host='localhost'):
    """
    Returns incoming features
    """
    ftrs = {}

    # # preproccess track
    # track = track.drop_duplicates('PointDate', keep='last')
    #
    # # smooth speed
    # pre_samp = speed.resample('S').interpolate(method='pchip')
    # resampled = pre_samp.rolling(window=10, center=True, win_type='parzen'). \
    #     mean().fillna(method='ffill').fillna(method='bfill')
    # smoothed_speed = resampled[track.PointDate].as_matrix()
    # track.Speed = smoothed_speed

    track = track[track.PointDate > pd.datetime(2015, 1, 1)]
    track = ways.delete_outliers_from_track(track)

    speed = track.Speed.copy()
    speed.index = track.PointDate
    total_secs = get_total_secs(track)

    # add_phone_cols(track)

    if len(track) == 0:
        return dict_to_frame({})
    car_track, ped_track = car_pedestrian_split(track)

    # 1,2
    speed_ftrs = get_speed_features(track, car_track)
    ftrs.update(speed_ftrs)

    # 3. add total track time and distance
    ftrs['total_secs'] = total_secs
    ftrs['total_dist'] = max(track.TotalMeters)

    # 4. stop statistics
    stop_ftrs, track_dif = get_stop_ftrs(car_track)
    ftrs.update(stop_ftrs)

    # 5. move statistics
    move_ftrs = get_move_ftrs(car_track, track_dif)
    ftrs.update(move_ftrs)

    # 7. Telephone usage
    '''
    a - angular velocity threshold rad/s
    s - speed threshold
    '''
    #    a, s = 1, 20
    #    sub_track = car_track[car_track.angular_velocity > a]
    #
    #    ftrs['low_speed_telephone_usage'] =  0 if len(sub_track) == 0 else\
    #        len(sub_track[sub_track.speed < s])/len(sub_track)
    #
    #    ftrs['high_speed_telephone_usage'] = 1 - ftrs['low_speed_telephone_usage']

    # 6. distance after car stops, pedestrian statistics
    ftrs['pedestrian_dist'] = max(ped_track.TotalMeters) - \
                              min(ped_track.TotalMeters)

    # 8. geo features analysis
    if use_geo_ftrs:
        import ai.geoftrs as gf
        # TODO: get geo features from smoothed track, not from raw. Impooves speed.
        if len(ped_track) > 0:
            ped_start = ped_track.iloc[0][['Latitude', 'Longitude']]
        else:
            ped_start = None
        geo_ftrs = gf.get_geoftrs(track, ped_start=ped_start, host=host)
        if len(geo_ftrs):
            geo_ftrs = geo_ftrs.iloc[0].to_dict()
        ftrs.update(geo_ftrs)

    ftrs_df = dict_to_frame(ftrs)
    return ftrs_df


def car_pedestrian_split(track, speed_max=9):
    """ drop last part, where user "goes home" after stop """

    track['pedestrian'] = (track.Speed <= speed_max).astype(int)
    first_dif = 1 if track.Speed.iloc[0] <= speed_max else -1
    track['ped_dif'] = track.pedestrian.diff().fillna(first_dif)
    ped = track[track['ped_dif'] == 1]
    car = track[track['ped_dif'] == -1]
    if len(ped) != 0 and len(car) != 0 and ped.index[-1] > car.index[-1]:
        last_index = ped.index[-1]
    else:
        last_index = track.index[-1]

    car_track = track.ix[:last_index].copy()
    ped_track = track.ix[last_index:].copy()

    return car_track, ped_track


def get_speed_features(track, car_track):
    ftrs = {}

    # 1. add speed distribution, standard deviation, median, mean    
    speed_bins = np.concatenate((np.arange(0, 90, 5), [100, 130, 160, 300]))
    speed_dist = get_hist_ftrs(car_track.Speed, speed_bins, 'v')
    ftrs.update(speed_dist)

    ftrs['speed_median'] = track.Speed.median()
    ftrs['speed_mean'] = track.Speed.mean()
    ftrs['speed_std'] = track.Speed.std()
    ftrs['speed_max'] = max(track.Speed)

    # 2. add acceleration distribution and standard deviation
    dv = track.Speed.diff().fillna(0)
    dt = track.PointDate.diff().fillna(0)
    dt = pd.to_timedelta(dt) / np.timedelta64(1, 's')

    # set 0 where time delta <= 0
    dv.loc[dt <= 0] = 0
    dt.loc[dt <= 0] = 1
    accs = dv / dt

    # delete rows where accs is greater than 7 standard deviations (remove
    # outliers)
    #    accs = accs[np.abs(accs - accs.mean()) <= 5 * accs.std()]

    acc_bins = np.arange(-23, 26, 3)
    acc_dist = get_hist_ftrs(accs, acc_bins, 'a')
    ftrs.update(acc_dist)

    pos_accs = accs[accs > 0]
    neg_accs = -accs[accs < 0]

    #    ftrs['acc_median'] = accs.median()
    ftrs['acc_mean'] = accs.mean()
    ftrs['acc_std'] = accs.std()

    ftrs['acc_pos_mean'] = pos_accs.mean()
    ftrs['acc_neg_mean'] = neg_accs.mean()
    ftrs['acc_pos_median'] = pos_accs.median()
    ftrs['acc_neg_median'] = neg_accs.median()
    ftrs['acc_pos_std'] = pos_accs.std()
    ftrs['acc_neg_std'] = neg_accs.std()

    return ftrs


def get_stop_ftrs(car_track):
    ftrs = {}

    car_track['inverse'] = (car_track.Speed <= 3).astype(int)
    first_dif = -1 if car_track.Speed.iloc[0] <= 3 else 1
    car_track['dif'] = car_track.inverse.diff().fillna(first_dif)

    dt = car_track[car_track.dif != 0].PointDate.diff().fillna(0)
    secs = dt / np.timedelta64(1, 's')
    secs = secs[secs > 5]  # only meaningful intervals
    track_dif = car_track.ix[secs.index]

    stop_dif = track_dif[track_dif.dif == -1]
    stop_secs = secs.ix[stop_dif.index]

    ts = get_total_secs(car_track)
    ftrs['rel_stand_time'] = 0 if ts == 0 else sum(stop_secs) / ts
    stop_secs = stop_secs[np.logical_and(stop_secs > 5, stop_secs < 10 * 60)]

    ftrs['nstops'] = len(stop_secs)
    ftrs['rel_stand_time_median'] = 0 if len(stop_secs) * ts == 0 else \
        np.median(stop_secs) / ts
    ftrs['rel_stand_time_std'] = 0 if len(stop_secs) * ts == 0 else \
        np.std(stop_secs) / ts

    stop_bins = np.array([5, 10, 25, 40, 60, 90, 2 * 60, 3 * 60, 10 * 60])
    stop_dist = get_hist_ftrs(stop_secs, stop_bins, 'st_t')
    ftrs.update(stop_dist)

    return ftrs, track_dif


def get_move_ftrs(car_track, track_dif):
    ftrs = {}

    ds = car_track[car_track.dif != 0].TotalMeters.diff().fillna(0)
    move_dif = track_dif[track_dif.dif == 1]
    move_len = ds.ix[move_dif.index]
    if sum(move_len) == 0:
        move_len = pd.Series([max(car_track.TotalMeters)])
    move_bins = np.array([10, 100, 250, 500, 1000, 1500, 2033, 3033, 4033,
                          10000, 20000, 30000, 60000, 180000])  # meters
    move_len = move_len[np.logical_and(move_len > min(move_bins),
                                       move_len < max(move_bins))]
    td = max(car_track.TotalMeters)
    ftrs['rel_move_len_median'] = 0 if len(move_len) * td == 0 else \
        np.median(move_len) / td
    ftrs['rel_move_len_std'] = 0 if len(move_len) * td == 0 else \
        np.std(move_len) / td

    move_dist = get_hist_ftrs(move_len, move_bins, 'mv_d')
    ftrs.update(move_dist)

    return ftrs


# ==============================================================================
# rich features  
# ==============================================================================

def get_rich_features(track):
    ftrs = {}
    track = track.copy()

    #    Height = track.Height
    #    track = track[np.abs(Height - Height.mean()) <= 5 * Height.std()]

    track.Height = track.Height.interpolate()
    numeric_dates = pd.to_numeric(track.PointDate)
    numeric_dates[track.PointDate.isnull()] = None
    track.PointDate = pd.to_datetime(numeric_dates.interpolate())
    track.fillna(0, inplace=True)

    ftrs['rel_from_device'] = 1 - np.bincount(track.FromSource)[0] / len(track.FromSource)
    ftrs['rel_urban'] = 1 - np.bincount(track.Urban)[0] / len(track.Urban)
    ftrs['height_min'] = min(track.Height)
    ftrs['height_max'] = max(track.Height)
    ftrs['height_mean'] = np.mean(track.Height)
    ftrs['height_median'] = np.median(track.Height)
    ftrs['height_std'] = np.std(track.Height)

    zero_limit = track.Speed[track.SpeedLimit == 0]
    overspeed = track.Speed / track.SpeedLimit
    overspeed.ix[zero_limit.index] = 1
    overspeed = overspeed[overspeed > 1]
    l = len(overspeed)
    ftrs['rel_overspeed_max'] = max(overspeed) if l else 1
    ftrs['rel_overspeed_mean'] = np.mean(overspeed) if l else 1
    ftrs['rel_overspeed_median'] = np.median(overspeed) if l else 1
    ftrs['rel_overspeed_std'] = np.std(overspeed) if l else 0

    start = min(track.PointDate)
    ftrs['day'] = start.day
    ftrs['weekday'] = start.weekday()
    ftrs['hour'] = start.hour
    ftrs['minute'] = start.minute

    ftrs_df = dict_to_frame(ftrs)
    return ftrs_df


# ==============================================================================
# info features
# ==============================================================================

def extract_track_info(track_info):
    # select columns as features
    usecols = ['Id', 'IncomingTrackId', 'StartDate', 'Distance',
               'Duration', 'AccelerationCount', 'DecelerationCount', 'Rating', 'PhoneUsage',
               'DistanceGPS', 'Urban', 'OverSpeedMileage',
               'MidOverSpeedMileage', 'HighOverSpeedMileage']  # 'CompanyId'
    track_info = track_info[usecols]

    track_info = track_info.copy()
    # add trips per day feature
    track_info.StartDate = pd.to_datetime(
        track_info.StartDate).apply(pd.Timestamp.date)
    trips_per_day = track_info[['StartDate']].groupby(
        ['StartDate']).size().reset_index()
    trips_per_day.rename(columns={0: 'trips_per_day'}, inplace=True)
    trips_per_day['trips_odd'] = trips_per_day.trips_per_day % 2
    track_info = track_info.merge(trips_per_day, on=['StartDate'])
    del track_info['StartDate']

    track_info.fillna(0, inplace=True)
    track_info.rename(columns={"Id": "TrackId"}, inplace=True)
    track_info.drop_duplicates(inplace=True)
    return track_info


# ==============================================================================
# misc
# ==============================================================================

def add_phone_cols(track):
    accs = track[['AccelerationX', 'AccelerationY', 'AccelerationZ']]
    track['R'] = np.linalg.norm(accs, axis=1)
    track['tetta'] = np.degrees(np.arccos(accs.AccelerationZ / track['R']))
    #    incoming_points['phi'] = np.degrees(
    #        np.arctan(accs.AccelerationY/accs.AccelerationX))
    track['angular_velocity'] = \
        np.sqrt(track['GyroscopeX'] ** 2 +
                track['GyroscopeY'] ** 2 +
                track['GyroscopeZ'] ** 2)
    track[['AccelerationX',
           'AccelerationY',
           'AccelerationZ']].fillna(0, inplace=True)


def get_distance_from_last_track(rich_points, last_track=None):
    if last_track is None:
        hist_points = rich_points
    else:
        hist_points = pd.concat([last_track, rich_points])

    grouped = hist_points.groupby(['TrackId'])

    first_df = grouped.first()[['PointDate', 'Longitude', 'Latitude']]
    first_df.sort_values(by='PointDate', inplace=True)
    first = first_df[['Latitude', 'Longitude']].values

    last_df = grouped.last()[['PointDate', 'Longitude', 'Latitude']]
    last_df.sort_values(by='PointDate', inplace=True)
    last = last_df[['Latitude', 'Longitude']].values

    zipped = zip(first[1:], last[:-1])
    shifts = [1000 * haversine(p1, p2) for p1, p2 in zipped]
    if len(shifts) == 0:
        shifts = [[0]]
    else:
        shifts.insert(0, np.median(shifts))
    shifts_df = pd.DataFrame(shifts, index=first_df.index,
                             columns=['dist_from_last_track'])
    return shifts_df


def get_end_points(track):
    #    begin = track.iloc[0]
    end = track.iloc[-1]
    return np.array([end.Longitude, end.Latitude])


def get_end_counts(smoothed):
    radii = [500, 1000]
    big_rad = 100 * 1000

    def closest(row, radius):
        return (sum(row < radius * tm.meter) - 1) / len(row)

    grouped = smoothed.groupby('TrackId')
    end = grouped.apply(get_end_points)
    X = end.values.tolist()
    dist_matrix = euclidean_distances(X, X)
    counts = pd.DataFrame([[closest(row, r) for r in [100, big_rad] + radii]
                           for row in dist_matrix], index=end.index,
                          columns=['rel_end_count_%s' % r
                                   for r in [100, big_rad] + radii])
    for r in radii:
        copied = counts['rel_end_count_%s' % r].copy()
        copied.loc[copied <= 0] = 1
        counts['end_count_100/%s' % r] = \
            counts['rel_end_count_100'] / copied
    # del counts['end_count_%s'%r]
    #    del counts['end_count_100']
    return counts


def get_total_secs(track):
    """
    Returns total number of seconds in track
    """
    if len(track):
        ts = np.timedelta64(
            max(track.PointDate) - min(track.PointDate)) / np.timedelta64(1, 's')
    else:
        ts = 0

    return ts


def get_hist_ftrs(a, bins, label):
    if len(a) == 0:
        hist = 0 * bins
    else:
        hist = np.histogram(a, bins=bins, density=True)[0]
    zipped = zip(bins, bins[1:], hist)
    dist = {'%s(%s,%s)' % (label, a, b): s for a, b, s in zipped}
    return dist


def natural_keys(text: str):
    """
    :type text: str
    """
    def atoi(_text):
        is_digit = True
        try:
            int(_text)
        except ValueError:
            is_digit = False
        return int(_text) if is_digit else _text

    return [atoi(c) for c in re.split('([+-]?\d+)', text)]


def dict_to_frame(d):
    df = pd.DataFrame.from_dict(d, orient='index').T
    cols = list(df.columns)
    cols.sort(key=natural_keys)
    df = df[cols]
    return df
