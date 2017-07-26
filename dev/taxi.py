import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from haversine import haversine
from matplotlib import pyplot as plt

import ai.features as ft
import ai.ways as ways
import dev.giro as giro
import dev.rotate_axes as rax
import streem.processing as pr
import test.test as test

SPEED_MAX = 9


def taxi_features(track):
    track = ways.delete_outliers_from_track(track)
    if len(track) == 0:
        return pd.DataFrame()
    theta_f = theta_features(track)
    z_f = rax.get_z_features(track)

    _, ped_track = ft.car_pedestrian_split(track)
    p_ftrs = parking_ftrs(ped_track)

    ftrs = theta_f.copy()
    for i, val in enumerate(z_f):
        ftrs['z_f%s' % i] = val

    for c in p_ftrs.columns:
        ftrs[c] = p_ftrs[c]

    return ftrs.to_frame().T


def theta_features(track):
    track.index = track.PointDate.values
    track = giro.add_phone_cols(track)
    pos = giro.theta_positions(track, max_amp=10, min_secs=5 * 60,
                               max_residual_std=8)

    pos_sub = pos.drop(['start', 'end'], axis=1)

    # take longest position
    longest = pos_sub.iloc[pos_sub['time_delta'].argmax()]

    return longest


def parking_ftrs(ped_track):
    # take only first 10 points of pedestrian track
    # the hope is that it will be a parking moment
    ped = ped_track[['Longitude', 'Latitude']].values
    angles = [angle_between(p1 - p2, p2 - p3) for
              p1, p2, p3 in zip(ped[:-1], ped[1:], ped[2:])]

    lens = [1000 * haversine(p1, p2) for p1, p2 in zip(ped[:-1], ped[1:])]
    lens = [l for l in lens if l < 100]
    count = 0
    max_count = 0
    for l in lens:
        if 0.5 < l < 30:
            count += 1
        else:
            max_count = max(max_count, count)
            count = 0

    # if len(angles) == 0:
    #     angles = [-1]
    # if len(lens) == 0:
    #     lens = [-1]

    # ftrs = pd.DataFrame([[min(angles), max(angles),
    #                       np.mean(angles), np.median(angles),
    #                       len(ped_track), np.std(angles),
    #                       min(lens), max(lens),
    #                       np.mean(lens), np.median(lens),
    #                       np.std(lens)
    #                       ]],
    #                     columns=['min_ped_angle', 'max_ped_angle',
    #                              'mean_ped_angle', 'median_ped_angle',
    #                              'len_ped', 'std_ped_angle',
    #                              'min_ped_len', 'max_ped_len',
    #                              'mean_ped_len', 'median_ped_len',
    #                              'std_ped_len']
    #                     )

    return angles, lens, sum(lens), len(lens), max_count


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    angle_between((1, 0, 0), (1, 0, 0))
    0.0
    angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def cross_validate(params, X, y):
    num_round = 4
    dtrain = xgb.DMatrix(X, y)
    res = xgb.cv(params, dtrain, num_round, nfold=5,
                 metrics=['auc'], seed=0, stratified=True, as_pandas=True)
    return res


def park_dist(inc_points):
    global SPEED_MAX
    grouped = inc_points.groupby('IncomingTrackId')
    angles, lens, lens0, nums, max_counts = [], [], [], [], []
    for _id, g in grouped:
        track = ways.delete_outliers_from_track(g.copy())
        _, ped_track = ft.car_pedestrian_split(track, speed_max=SPEED_MAX)
        a, l, l0, n, m = parking_ftrs(ped_track)
        angles += a
        lens += l
        lens0 += [l0]
        nums += [n]
        max_counts += [m]
    return angles, lens, lens0, nums, max_counts


def park_dist_worker(zipped_users):
    angles_t, lens_t, lens_t_0, nums_t, max_counts_t, \
    angles_o, lens_o, lens_o_0, nums_o, max_counts_o = [], [], [], [], [], [], [], [], [], []

    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        print(path_to_tracks)
        incoming_points = pr.read_points(path_to_incoming)
        tracks = pd.read_csv(path_to_tracks)

        inc_original, inc_taxi = sample_taxi_original(incoming_points, tracks)

        a, l, l0, n, m = park_dist(inc_taxi)
        angles_t += a
        lens_t += l
        lens_t_0 += l0
        nums_t += n
        max_counts_t += m

        a, l, l0, n, m = park_dist(inc_original)
        angles_o += a
        lens_o += l
        lens_o_0 += l0
        nums_o += n
        max_counts_o += m

    return angles_t, lens_t, lens_t_0, nums_t, max_counts_t, \
           angles_o, lens_o, lens_o_0, nums_o, max_counts_o


def plot_park(inc_points, name, save_dir):
    global SPEED_MAX
    grouped = inc_points.groupby('IncomingTrackId')
    for _id, g in grouped:
        track = ways.delete_outliers_from_track(g.copy())
        car_track, ped_track = ft.car_pedestrian_split(track, speed_max=SPEED_MAX)
        car_track = car_track[car_track['PointDate'] >= car_track['PointDate'] -
                              np.timedelta64(60 * 2, 's')]
        if (len(car_track['Longitude'].unique()) > 2) & \
                (len(car_track['Latitude'].unique()) > 2):
            plt.close("all")
            lat, lon = car_track['Latitude'], car_track['Longitude']
            _max = max(max(lat) - min(lat), max(lon) - min(lon))
            f, ax = plt.subplots()
            f.set_size_inches(9.86, 9.86)
            car_track.plot('Longitude', 'Latitude', ax=ax,
                           xlim=(min(lon), min(lon) + _max),
                           ylim=(min(lat), min(lat) + _max))
            last = car_track.iloc[-1:]
            plt.scatter(last['Longitude'], last['Latitude'], color='red', axes=ax)
            f.tight_layout()
            f.savefig(save_dir + '%s_%s.png' % (name, _id))


def plot_park_wrapper(save_dir, zipped_users):
    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        print(path_to_tracks)
        incoming_points = pr.read_points(path_to_incoming)
        tracks = pd.read_csv(path_to_tracks)

        inc_original, inc_taxi = sample_taxi_original(incoming_points, tracks)

        plot_park(inc_taxi, 'taxi', save_dir)
        plot_park(inc_original, 'orig', save_dir)


def thetta_plot_walker(incoming_points, save_dir, name):
    global SPEED_MAX
    grouped = incoming_points.groupby('IncomingTrackId')
    for _id, track in grouped:
        if sum(track['AccelerationZOriginal']) == 0:
            continue
        track = ways.delete_outliers_from_track(track)
        track.index = track.PointDate.values
        giro.add_phone_cols(track)
        if len(track) == 0:
            continue
        _, ped_track = ft.car_pedestrian_split(track, speed_max=SPEED_MAX)
        if len(ped_track) < 3:
            continue
        if len(ped_track['theta'].unique()) < 2:
            continue

        plt.close("all")
        f, ax = plt.subplots()
        f.set_size_inches(18.67, 9.86)
        ped_track.plot('PointDate', 'theta', ax=ax)
        ped_track.plot('PointDate', 'smoothed_theta', ax=ax)
        f.tight_layout()
        f.savefig(save_dir + '%s_%s.png' % (name, _id))


def thetta_plot_walker_wrapper(save_dir, zipped_users):
    global SPEED_MAX
    for path_to_tracks, path_to_rich, path_to_incoming in zipped_users:
        print(path_to_tracks)
        incoming_points = pr.read_points(path_to_incoming)
        tracks = pd.read_csv(path_to_tracks)

        tracks = tracks[tracks['TrackOrigin'] == 'OriginalDriver']
        incoming_points = incoming_points[incoming_points['IncomingTrackId'].isin(tracks['IncomingTrackId'])]

        # inc_original, inc_taxi = sample_taxi_original(incoming_points, tracks)

        # thetta_plot_walker(inc_taxi, save_dir, 'taxi')
        thetta_plot_walker(incoming_points, save_dir, 'original')


def sample_taxi_original(incoming_points, tracks):
    taxi = tracks[tracks['TrackOrigin'] == 'Taxi'].IncomingTrackId.values
    original = tracks[tracks['TrackOrigin'] == 'OriginalDriver'].IncomingTrackId.values
    original = original[:len(taxi)]
    inc_taxi = incoming_points[incoming_points.IncomingTrackId.isin(taxi)]
    inc_original = incoming_points[incoming_points.IncomingTrackId.isin(original)]
    return inc_original, inc_taxi


if __name__ == '__main__':
    data_dir = '/home/guyos/Yandex.Disk/raxel/data/test/'
    zipped_users = test.zipped_tripples(data_dir, n1='tracks', n2='rich', n3='incoming')

    save_dir = '/home/guyos/Downloads/varlam/'

    # cpu_cores = mp.cpu_count()
    # pool = mp.Pool(cpu_cores)
    # func = partial(thetta_plot_walker_wrapper, save_dir)
    # result = pool.map(func, np.array_split(zipped_users, cpu_cores))
    # pool.close()
    # pool.join()

    cpu_cores = mp.cpu_count()
    pool = mp.Pool(cpu_cores)
    func = partial(plot_park_wrapper, save_dir)
    result = pool.map(func, np.array_split(zipped_users, cpu_cores))
    pool.close()
    pool.join()

    ########################################################################


    taxi_pass = pd.read_csv('/home/guyos/Yandex.Disk/raxel/data/taxi/taxi_pass.csv.gz')
    taxi_pass_all = pd.read_csv('/home/guyos/Yandex.Disk/raxel/data/taxi/taxi_pass_all.csv.gz')

    token = '3a2de908-215e-4a2c-b5d4-2db895f126bf'.lower()

    concx = pd.concat([dots[['StartLon', 'FinishLon']],
                       dots_all[['StartLon', 'FinishLon']]]).values.flatten()
    concy = pd.concat([dots[['StartLat', 'FinishLat']],
                       dots_all[['StartLat', 'FinishLat']]]).values.flatten()
    xmin = min(concx)
    xmax = max(concx)
    ymin = min(concy)
    ymax = max(concy)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    for _, row in dots.iterrows():
        ax.scatter(row.StartLon, row.StartLat, color='green', alpha=0.4)
        ax.scatter(row.FinishLon, row.FinishLat, color='yellow', alpha=0.4)
        ax.plot([row.StartLon, row.FinishLon],
                [row.StartLat, row.FinishLat],
                alpha=0.1,
                color='red')

    plt.xlim((xmin-0.01, xmax+0.01))
    plt.ylim((ymin-0.01, ymax+0.01))
    fig1.tight_layout()
    plt.show()

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    orig = dots_all[~dots_all['RichTrackId'].isin(dots.Id)]
    for _, row in orig.iterrows():
        ax.scatter(row.StartLon, row.StartLat, color='green', alpha=0.4)
        ax.scatter(row.FinishLon, row.FinishLat, color='yellow', alpha=0.4)
        ax.plot([row.StartLon, row.FinishLon],
                [row.StartLat, row.FinishLat],
                alpha=0.1,
                color='blue')

    plt.xlim((xmin-0.01, xmax+0.01))
    plt.ylim((ymin-0.01, ymax+0.01))
    fig1.tight_layout()
    plt.show()

