import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import ai.ways as ways
import streem.processing as pr
from matplotlib import pyplot as plt

import test.test as test
import multiprocessing as mp
from functools import partial

import os


def get_distances(track, geo_points):
    coords = track[['Longitude', 'Latitude']].values
    line = LineString(coords)
    geo_line = gpd.GeoSeries(line, crs={'init': 'epsg:4326'})
    geo_line = geo_line.to_crs({'init': 'epsg:3857'})

    geom_track = track[['Longitude', 'Latitude']].apply(lambda x: Point(x), axis=1)
    geo_track = gpd.GeoDataFrame(track, crs={'init': 'epsg:4326'}, geometry=geom_track)
    geo_track = geo_track.to_crs({'init': 'epsg:3857'})

    geo_track['dist'] = geo_track.geometry.apply(lambda x: geo_line.project(x))

    poly = geo_line.buffer(35)
    cond = geo_points.intersects(poly.ix[0])
    sub_points = geo_points[cond].copy()
    sub_points['dist'] = sub_points.geometry.apply(lambda x: geo_line.project(x))

    return geo_track, sub_points


def fit_polynom(points_result):
    sorted_df = points_result.sort_values(by='dist')
    x = sorted_df['dist'].values
    y = sorted_df['Speed']
    z = np.polyfit(x, y, deg=10)
    f = np.poly1d(z)

    x_new = np.linspace(x[0], x[-1], np.int64(max(x) / 25.0))
    y_new = f(x_new)

    return x_new, y_new


def speed_dist(track, points):
    lat_min, lat_max = track['Latitude'].min(), track['Latitude'].max()
    lon_min, lon_max = track['Longitude'].min(), track['Longitude'].max()

    points_box = points[points.Latitude.between(lat_min, lat_max) &
                        points.Longitude.between(lon_min, lon_max)]

    if len(points_box) == 0:
        return None, None

    geom_points = points_box[['Longitude', 'Latitude']].apply(lambda x: Point(x), axis=1)
    geo_points = gpd.GeoDataFrame(points_box, crs={'init': 'epsg:4326'}, geometry=geom_points)
    geo_points = geo_points.to_crs({'init': 'epsg:3857'})

    # poly = geo_line.buffer(5)
    # ax = poly.plot()
    # geo_points.plot(ax=ax)

    grouped = track.groupby('labels')
    start_dist = 0
    ltrack, lpoints = [], []
    for label, sub_track in grouped:
        geo_track, sub_points = get_distances(sub_track, geo_points)
        geo_track['dist'] += start_dist
        sub_points['dist'] += start_dist
        start_dist = max(geo_track['dist'])
        ltrack.append(pd.DataFrame(geo_track))
        lpoints.append(pd.DataFrame(sub_points))

    track_result = pd.concat(ltrack)
    points_result = pd.concat(lpoints)

    return track_result, points_result


def log_empty(fname):
    with open(fname, 'w') as f:
        f.write('empty tarck')


def proceed_track(test_id, pics_dir, incoming_points, original):
    # test_id = labeled.iloc[6]
    points = incoming_points[incoming_points.IncomingTrackId.isin(original)]
    track = incoming_points[incoming_points.IncomingTrackId == test_id]
    track = ways.delete_outliers_from_track(track)

    ###########################
    track_result, points_result = speed_dist(track, points)
    if points_result is None or len(points_result) == 0:
        log_empty(pics_dir + str(test_id) + '.txt')
        return
    # x_new, y_new = fit_polynom(points_result)
    ###########################

    plt.close("all")

    fig, ax = plt.subplots()
    fig.set_size_inches(18.67, 9.86)

    points_result.plot.scatter('dist', 'Speed', alpha=0.3, ax=ax,
                               title='original tracks + passanger')
    track_result.plot('dist', 'Speed', ax=ax, color='orange', alpha=0.9, label='%s %s' % (label, test_id))
    # ax.plot(x_new, y_new)
    ax.set_xlabel('distance')
    fig.tight_layout()
    if len(points_result) > 0 and len(track_result) > 0:
        plt.xlim(max(min(points_result['dist']), min(track_result['dist'])),
                 min(max(points_result['dist']), max(track_result['dist'])))
    fig.savefig(pics_dir + '%s.png' % test_id)

    # ax1 = track.sort_values(by='PointDate', ascending=False).plot('Longitude', 'Latitude')
    # points_result[['Longitude', 'Latitude']].plot.scatter('Longitude', 'Latitude', ax=ax1)
    # more than 3 tracks in bin
    # track.to_csv('~/tr.csv')


def proceed_track_worker(incoming_points, pics_dir, original, inc_ids):
    for test_id in inc_ids:
        proceed_track(test_id, pics_dir, incoming_points, original)
        print(test_id)


def gen_plots(pics_dir, path_to_tracks, path_to_incoming, label):
    tracks = pd.read_csv(path_to_tracks)
    original = tracks[tracks['TrackOrigin'] == 'OriginalDriver'].IncomingTrackId
    labeled = tracks[tracks['TrackOrigin'] == label].IncomingTrackId

    incoming_points = pr.read_points(path_to_incoming,
                                     usecols=['IncomingTrackId',
                                              'Longitude',
                                              'Latitude',
                                              'Speed',
                                              'PointDate'])

    valid_ids = np.concatenate([original, labeled])
    incoming_points = incoming_points[incoming_points['IncomingTrackId'].isin(valid_ids)]
    incoming_points.sort_values(['PointDate', 'IncomingTrackId'], inplace=True)
    incoming_points.drop_duplicates(['Latitude', 'Longitude', 'IncomingTrackId'], keep='last', inplace=True)
    incoming_points.drop_duplicates(['PointDate', 'IncomingTrackId'], keep='last', inplace=True)

    proceed_track_worker(incoming_points, pics_dir, original, np.squeeze(labeled.values))

    # nproc = min(mp.cpu_count(), len(labeled))
    # pool = mp.Pool()
    # func = partial(proceed_track_worker, incoming_points, pics_dir, original)
    # result = pool.map(func, np.array_split(np.squeeze(labeled.values), nproc))
    # pool.close()
    # pool.join()


def gen_plots_worker(pics_dir, label, zipped_users):
    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        gen_plots(pics_dir, path_to_tracks, path_to_incoming, label)


def calc_bins(track, bin_len=50):
    1


def split_by_size(zipped_users, cpu_cores):
    new_zipped = []
    for tup in zipped_users:
        tup += (os.path.getsize(tup[2]),)
        new_zipped.append(tup)
    zipped_df = pd.DataFrame(new_zipped)
    zipped_df.sort_values(by=3, inplace=True, ascending=False)

    vals = zipped_df[[0,1,2]].values
    arrays = [[] for _ in range(cpu_cores)]
    for i, val in enumerate(vals):
        l = arrays[i%cpu_cores]
        l.append(val)

    return arrays


if __name__ == '__main__':

    data_dir = '/home/guyos/Yandex.Disk/raxel/data/passanger/'
    pics_dir = data_dir + 'plots/'

    zipped_users = test.zipped_tripples(data_dir, n1='tracks', n2='rich', n3='incoming')

    # gen_plots_worker(pics_dir, zipped_users)

    cpu_cores = mp.cpu_count()
    pool = mp.Pool(cpu_cores)
    func = partial(gen_plots_worker, pics_dir, 'Passanger')
    result = pool.map(func, split_by_size(zipped_users, cpu_cores))
    pool.close()
    pool.join()


