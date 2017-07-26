"""
                   Алгоритм определения такси в/из аэропорта

Общая идея: если человек приехал в аэропорт (или уехал из аэропорта), то мы смотрим
     а) где начинается следующий трек. Если он начинается более чем в 400 км от конца последнего трека, значит,
        человек улетел
     б) если следующий трек начинается близко к концу предыдущего трека, но через длительное время,
        тоже считаем, что человек улетал (и не использовал приложение в роуминге)

Подробности:

0. Анализируемый трек изначально очищается - сортируется по PointDate, а затем удаляются различные дубликаты.
   Если это делается в общем алгоритме, то функцию clear_track применять не нужно.

1. Координаты аэропорта берем из файла AirportUsers.csv (там находятся координаты всех аэропортов мира).

2. Человек мог приехать близко к аэропорту, но не прямо к самому центру (к точке, указанной в файле).
   В этом случае мы не можем быть уверены, что человек приехал именно в аэропорт, возможно, у него дача или дом где-то
   рядом (например, так у Димы Рудаша). В таком случае мы смотрим, насколько далеко человек остановился от центральной
   точки аэропорта и в зависимости от этого решаем, улетел человек или нет (в алгоритме для этого введены константы
   AIRPORT и NEAR_AIRPORT).

   There exists very big airports: from the center of airport to end point of track may be distance about 3000-3500 m
   (for example, Domodedovo airport). But 3500 m is a big distance and there are tracks not to airport with same
   distance. That's why we should take additional parameter (BIG_DELTA_TIME and VERY_BIG_DELTA_TIME) to distinguish
   tracks to airport and tracks to other place.

3. Для треков вне домашней зоны надо бы проверить, являются ли они треками на автобусе, поезде или ходьбой. Если нет,
   то можно говорить, что трек на такси. Домашняя зона определяется как медианная точка всех треков водителя.
"""

import os
import pickle

import gpxpy.geo as mod_geo
import numpy as np
import pandas as pd
import redis
from dateutil import parser

from ai.geoftrs import get_data_dir

r = redis.Redis(host='localhost')

AIRPORT = 2
NEAR_AIRPORT = 1
NOT_AIRPORT = 0
BIG_DISTANCE = 0
BIG_DELTA_TIME = 2
VERY_BIG_DELTA_TIME = 1
SMALL_DELTA_TIME = 0
HOME = 1
NOT_HOME = 0
DAY = 24


def cache_csv(tag, fname):
    blob = r.get(tag)
    if blob is None:
        fname = os.path.join(get_data_dir(), fname)
        df = pd.read_csv(fname)
        r.set(tag, pickle.dumps(df))
    else:
        df = pickle.loads(blob)

    return df


def find_airport_in_neigbourhood(point, airports):
    """Calculates dist: distance from point to nearest airport
    If dist <= 1500 then its definitely airport;
    if dist in (1500; 3500] then it may be airport;
    else it is not airport.
    """
    zipped = zip(airports['latitude'], airports['longitude'])
    L = np.min([mod_geo.haversine_distance(la, lo, point[0], point[1]) for la, lo in zipped])

    if L <= 1500:
        return AIRPORT, L
    elif L <= 3500:
        return NEAR_AIRPORT, L
    return NOT_AIRPORT, L


def clear_track(track):
    """
    Clear track: sort it and remove duplicates
    """
    track = track.copy()
    track.sort_values('PointDate', inplace=True)
    track.drop_duplicates(['Latitude', 'Longitude'], keep='last', inplace=True)
    track.drop_duplicates(['PointDate'], keep='last', inplace=True)
    track.reset_index(inplace=True)

    return track


def check_airport(start_end, airports):
    """
    Check if start (or end) point of track coincides with airport.
    Also return dates and coords of start and end points.
    """
    start_date, start_lat, start_lon, end_date, end_lat, end_lon = start_end

    start_coords = (start_lat, start_lon)
    res_start, L_start = find_airport_in_neigbourhood(start_coords, airports)

    end_coords = (end_lat, end_lon)
    res_end, L_end = find_airport_in_neigbourhood(end_coords, airports)

    if max(L_start, L_end) <= 3500:
        if L_start < L_end:
            res_end = 0
        else:
            res_start = 0

    date = (start_date, end_date)
    coords = (start_coords, end_coords)
    res = (res_start, res_end)

    return date, coords, res


def check_distance(pt_first, pt_next):
    """
    return 0 if distance is bigger than 400 km
    else return 1
    """

    L = mod_geo.haversine_distance(pt_first[1], pt_first[0], pt_next[1], pt_next[0])

    if L >= 400000:
        return 0
    return 1


def check_time(date_first, date_next):
    """
    Return category of delta time between date_first and date_next
    (very big, big, small delta time).
    """
    delta_time = date_first.timestamp() - date_next.timestamp()

    if delta_time > 3600 * 5 * DAY:
        return VERY_BIG_DELTA_TIME
    elif delta_time > 3600 * 2 * DAY:
        return BIG_DELTA_TIME

    return SMALL_DELTA_TIME


def get_avg_points(device_token, engine):
    qtext = '''
    SELECT [RichTrackId], rt.[prediction]
          ,[AvgLat] as Latitude
          ,[AvgLon] as Longitude
    FROM [MobileServiceStage].[dbo].[RichTrackDetails] as rtd
    inner join [MobileServiceStage].[dbo].[RichTracks]  as rt
    on rtd.[RichTrackId] = rt.[Id] and rtd.[DeviceToken] = ?
    order by rt.[StartDate]
    '''

    end_points = pd.read_sql(qtext, engine, params=[device_token])

    return end_points


def compute_track_start_end(track):
    start_date = track.iloc[0]['PointDate']
    start_lat, start_lon = track.iloc[:3][['Latitude', 'Longitude']].values.mean(axis=0)

    end_date = track.iloc[-1]['PointDate']
    end_lat, end_lon = track.iloc[-3:][['Latitude', 'Longitude']].values.mean(axis=0)
    return start_date, start_lat, start_lon, end_date, end_lat, end_lon


def query_track_start_end(rich_track_id, engine):
    qtext = '''
    SELECT
          rt.StartDate as start_date
          ,rtd.[StartLat] as start_lat
          ,rtd.[StartLon] as start_lon
          ,rt.EndDate as end_date
          ,rtd.[FinishLat] as end_lat
          ,rtd.[FinishLon] as end_lon
      FROM [MobileServiceStage].[dbo].[RichTrackDetails] as rtd (nolock)
      inner join [MobileServiceStage].[dbo].[RichTracks] as rt (nolock)
      on rt.Id = ? and rtd.[RichTrackId] = rt.Id        
      '''

    df = pd.read_sql(qtext, engine, params=[int(rich_track_id)])
    if len(df) == 0:
        return None
    else:
        start_date, start_lat, start_lon, end_date, end_lat, end_lon = df.iloc[0].values
        start_date = parser.parse(start_date)
        end_date = parser.parse(end_date)
        return start_date, start_lat, start_lon, end_date, end_lat, end_lon


def home_zone(track, home_coords):
    Res = HOME

    if isinstance(track, tuple) == True:
        coords = (track[0], track[1])
    else:
        coords = (track.iloc[0]['Latitude'], track.iloc[0]['Longitude'])

    if mod_geo.haversine_distance(
            coords[0], coords[1], home_coords[0], home_coords[1]) >= 400000:
        Res = NOT_HOME

    return Res


def is_taxi_airport_old(track, engine):
    """
    Old main algorithm.
    returns: 0 - original driver, 1 - taxi, -1 - other
    """
    track = clear_track(track)
    if len(track) < 10:
        # по 10-ти точкам ничего не скажешь
        return 0

    start_end = compute_track_start_end(track)

    # TODO: поместить файл с аэропортами в Redis для скорости
    airports = cache_csv('all_airports', 'all_airports.csv')
    date_curr, coords_curr, (start, end) = check_airport(start_end, airports)

    device_token = track['DeviceToken'].iloc[0]
    rich_track_id = track['RichTrackId'].iloc[0]

    tracks_by_date = get_avg_points(device_token, engine)
    home = np.median(np.round(tracks_by_date['Latitude'].values, 1)), \
           np.median(np.round(tracks_by_date['Longitude'].values, 1))

    if start == NOT_AIRPORT and end == NOT_AIRPORT:
        return 0 if home_zone(track, home) == HOME else -1

    ix = tracks_by_date[tracks_by_date['RichTrackId'] == rich_track_id].index[0]

    rich_col = tracks_by_date['RichTrackId']
    if start != NOT_AIRPORT and ix != tracks_by_date.index[0]:
        pt_start, res_curr, neigb_id = True, start, rich_col.ix[ix - 1]
    elif end != NOT_AIRPORT and ix != tracks_by_date.index[-1]:
        pt_start, res_curr, neigb_id = False, end, rich_col.ix[ix + 1]
    else:
        # if it is first track of user and it is from airport or
        # it is last track of user and it is to airport
        return -1 if (start == AIRPORT or end == AIRPORT) else 0

    start_end = query_track_start_end(neigb_id, engine)
    date_neigb, coords_neigb, res_neigb = check_airport(start_end, airports)

    # set params for two cases: if start point of track is airport
    # or if end point of track is airport.
    first_point, second_point = (coords_neigb[1], coords_curr[0]) if pt_start else \
        (coords_curr[1], coords_neigb[0])
    first_date, second_date = (date_neigb[1], date_curr[0]) if pt_start else \
        (date_curr[1], date_neigb[0])

    if check_distance(first_point, second_point) == BIG_DISTANCE:
        return 1
    else:
        delta_time = check_time(first_date, second_date)
        if delta_time == SMALL_DELTA_TIME:
            return 0 if home_zone(track, home) == HOME else -1
        elif (res_curr == AIRPORT and delta_time == BIG_DELTA_TIME) or \
                (res_curr == NEAR_AIRPORT and delta_time == VERY_BIG_DELTA_TIME):
            return 1

    return -1


def is_taxi_airport(track, conn):
    """
    Main algorithm.
    returns: 0 - original driver, 1 - taxi, -1 - other
    """
    track = clear_track(track)
    start_end = compute_track_start_end(track)

    # TODO: поместить файл с аэропортами в Redis для скорости
    airports = cache_csv('all_airports', 'all_airports.csv')
    date_curr, coords_curr, (start, end) = check_airport(start_end, airports)

    device_token = track['DeviceToken'].iloc[0]
    rich_track_id = track['RichTrackId'].iloc[0]

    tracks_by_date = get_avg_points(device_token, conn)
    home = np.median(np.round(tracks_by_date['Latitude'].values, 1)), \
           np.median(np.round(tracks_by_date['Longitude'].values, 1))

    if start == NOT_AIRPORT and end == NOT_AIRPORT and \
                    home_zone(track, home) == HOME:
        return 0

    ix = tracks_by_date[tracks_by_date['RichTrackId'] == rich_track_id].index[0]
    rich_col_ordered = tracks_by_date['RichTrackId']

    # if start is airport or near airport
    if start != NOT_AIRPORT:
        if ix != tracks_by_date.index[0]:
            pt_start, res_curr, neigb_id = True, start, rich_col_ordered.iloc[ix - 1]
        elif start == AIRPORT:
            return -1
        else:
            return 0
    # if end is airport or near airport
    elif end != NOT_AIRPORT:
        if ix != tracks_by_date.index[-1]:
            pt_start, res_curr, neigb_id = True, end, rich_col_ordered.iloc[ix + 1]
        elif end == AIRPORT:
            return -1
        else:
            return 0
    # if not airport, but not home: it is new part!
    elif home_zone(track, home) != HOME and ix != tracks_by_date.index[0]:
        prev_id = rich_col_ordered.iloc[ix - 1]
        _, _, _, _, end_lat, end_lon = query_track_start_end(prev_id, conn)
        if home_zone((end_lat, end_lon), home) != HOME and \
                        tracks_by_date[tracks_by_date.RichTrackId == prev_id]['prediction'].values[
                            0] == 'OriginalDriver' and \
                        mod_geo.haversine_distance(
                            track.iloc[0]['Latitude'], track.iloc[0]['Longitude'], end_lat, end_lon) < 100000:
            return 0
        return -1

    start_end = query_track_start_end(neigb_id, conn)
    date_neigb, coords_neigb, res_neigb = check_airport(start_end, airports)

    # set params for two cases: if start point of track is airport
    # or if end point of track is airport.
    first_point, second_point = (coords_neigb[1], coords_curr[0]) if pt_start else \
        (coords_curr[1], coords_neigb[0])
    first_date, second_date = (date_neigb[1], date_curr[0]) if pt_start else \
        (date_curr[1], date_neigb[0])

    if check_distance(first_point, second_point) == BIG_DISTANCE:
        return 1
    else:
        delta_time = check_time(first_date, second_date)
        if delta_time == SMALL_DELTA_TIME:
            return 0 if home_zone(track, home) == HOME else -1
        elif (res_curr == AIRPORT and delta_time == BIG_DELTA_TIME) or \
                (res_curr == NEAR_AIRPORT and delta_time == VERY_BIG_DELTA_TIME):
            return 1

    return -1


if __name__ == '__main__':
    from sqlalchemy import create_engine

    server = ''
    user = ''
    pas = ''
    engine = create_engine("mssql+pyodbc://%s:%s@company-ab-sql/?trusted_connection=no" % (user, pas))

    device_token = '9fbbbd45-1e8d-4430-92b5-96f8fd119b7c'
    qtracks = '''
        SELECT Id
        FROM [MobileServiceStage].[dbo].[RichTracks] (nolock)
        where [DeviceToken] = ?    
    '''

    tracks = pd.read_sql(qtracks, engine, params=[device_token])

    qrich = '''
        SELECT *
        FROM [MobileServiceStage].[dbo].[RichTrackPoints] (nolock)
        where [RichTrackId] = ?        
    '''

    import time

    res = []
    for rich_track_id in tracks['Id'].values:
        track = pd.read_sql(qrich, engine, params=[int(rich_track_id)],
                            parse_dates=['PointDate'])

        start = time.time()
        is_taxi = is_taxi_airport(track, engine)
        end = time.time()
        tdelta = end - start

        trip = (rich_track_id, is_taxi, tdelta)
        print('rich_track_id %s, is_taxi %s, time %s' % trip)
        res.append(trip)

    res = pd.DataFrame(res, columns=['rich_track_id', 'is_taxi', 'tdelta'])
    times = res['tdelta']
    print('total seconds %s, median time %s, max time %s' %
          (np.sum(times), np.median(times), np.max(times)))
    res[res['is_taxi'] == 1]
