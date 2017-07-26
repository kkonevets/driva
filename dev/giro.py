#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:18:45 2016

@author: Kirill Konevets
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import ai.ways as ways


def amplitude(array):
    return max(array) - min(array)


def add_phone_cols(track):
    accs = track[['AccelerationXOriginal', 'AccelerationYOriginal', 'AccelerationZOriginal']]
    track['R'] = np.linalg.norm(accs, axis=1)
    track['theta'] = np.degrees(np.arccos(accs.AccelerationZOriginal / track['R']))
    track.loc[track['theta'].isnull(), 'theta'] = 0

    # track['angular_velocity'] = np.linalg.norm(track[['GyroscopeXOriginal',
    #                                                   'GyroscopeYOriginal',
    #                                                   'GyroscopeZOriginal']], axis=1)
    # track[['AccelerationXOriginal', 'AccelerationYOriginal',
    #        'AccelerationZOriginal']].fillna(0, inplace=True)

    theta = track.theta.copy()
    pre_samp = theta.resample('S').interpolate(method='linear')
    resampled = pre_samp.rolling(window=200, center=True, win_type='parzen'). \
        mean().fillna(method='ffill').fillna(method='bfill')
    # resampled = pre_samp.ewm(span=200).mean().fillna(method='ffill').fillna(method='bfill')
    track['smoothed_theta'] = resampled[theta.index].values

    amps = resampled.rolling(window=30, center=True).apply(func=amplitude). \
        fillna(method='ffill').fillna(method='bfill')
    track['theta_amplitude'] = amps[theta.index].values

    return track


def median_cosine(ch):
    """
    Get cosine of angle between projected acceleration on XY and Y
    """
    median_ax = ch['AccelerationXOriginal'].median()
    median_ay = ch['AccelerationYOriginal'].median()
    mod = np.sqrt(median_ax ** 2 + median_ay ** 2)
    cosine = median_ay / mod if mod else 0

    return cosine


def max_eigenvec(ch):
    # get first component in (acc_z, acc)
    acc_z = ch['AccelerationZOriginal']
    acc = ch['Speed'].diff().fillna(0)
    X = np.array([acc, acc_z]).transpose()
    X -= X.mean(axis=0)
    if min(X.std(axis=0)) == 0:
        return [0, 0]
    X /= X.std(axis=0)
    # delete outliers
    # X = X[np.min(np.abs(X) <= 3*X.std(axis=0), axis=1)]

    pca = PCA(n_components=2)
    pca.fit_transform(X)
    evec = pca.components_[0]

    # from sklearn.mixture import BayesianGaussianMixture
    #
    # estimator = BayesianGaussianMixture(n_components=1,
    #                                     covariance_type='full')
    # estimator.mean_prior = ([0, 0])
    # estimator.fit(X)
    # estimator.predict(X)
    # covar = estimator.covariances_[0]
    #
    # v, w = np.linalg.eigh(covar)

    # import matplotlib.pyplot as plt
    # plt.scatter(x=X[:, 0], y=X[:, 1], alpha=0.3)
    # # plt.plot(range(-100, 100), 200 * [0])
    # # plt.plot(200 * [0], np.array(range(-100, 100)) / 40)
    # plt.plot([0, evec[0]], [0, evec[1]], color='red')

    return evec


def theta_positions(track, max_amp=4, min_secs=10 * 60,
                    max_residual_std=3.5):
    cols = ['start', 'end', 'time_delta',
            'theta_0', 'rstd', 'amp', 'screen_direction',
            'evec0', 'evec1']
    empty_pos = pd.DataFrame([[-1] * len(cols)], columns=cols)
    if len(track) == 0 or track['AccelerationXOriginal'].sum() == 0:
        return empty_pos

    track['inverse'] = (track['theta_amplitude'] <= max_amp).astype(int)
    first_dif = -1 if track.theta_amplitude.iloc[0] <= max_amp else 1
    last_dif = -1 if track.theta_amplitude.iloc[-1] <= max_amp else 1
    track['dif'] = track.inverse.diff()
    track.ix[0, 'dif'] = first_dif
    track.ix[-1, 'dif'] = last_dif

    intersects = track[track.dif != 0]

    valid_difs = intersects['dif']
    index_ranges = zip(valid_difs[:-1].index, valid_difs[1:].index)
    chunks = [track[(track.PointDate >= start) &
                    (track.PointDate < end)] for start, end in index_ranges]
    below_max_amp = [ch for ch in chunks if ch['inverse'].iloc[0]]

    lpos = []
    # ch = below_max_amp[2]
    for ch in below_max_amp:
        start, end = min(ch.PointDate), max(ch.PointDate)
        time_delta = (end - start) / np.timedelta64(1, 's')
        theta_0 = ch['smoothed_theta'].mean()
        residual = ch['smoothed_theta'] - theta_0
        rstd = residual.std()
        cosine = median_cosine(ch)
        amp = ch['theta_amplitude'].max()

        # y axis should not be upside down - median theta > pi/4
        if time_delta < min_secs or rstd > max_residual_std:  # or \
            # 0 <= cosine < 1 / 2 ** 0.5: # or not (95 < theta_0 < 120):
            continue
        # print(name, start, end, time_delta, theta_0, rstd, amp)

        evec = max_eigenvec(ch)

        screen_direction = evec[0] * evec[1]
        lpos.append((start, end, time_delta, theta_0, rstd, amp, screen_direction,
                     evec[0], evec[1]))

    if len(lpos):
        positions = pd.DataFrame(lpos, columns=cols)
    else:
        positions = empty_pos
    return positions


def stationary_positions(track, name, max_amp=4, min_secs=10 * 60,
                         max_residual_std=3.5):
    ftrs = theta_positions(track, max_amp, min_secs, max_residual_std)
    # eigen vector should be in first quarter - phone screen is
    # opposite to car motion
    positions = ftrs[ftrs['screen_direction'] > 0].copy()
    positions['name'] = name

    return positions


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import streem.processing as pr

    # fname = 'data_incoming.csv.gz'
    fname = '0dbe23cb-db3d-4f3a-ada1-a700f1e9b668_incoming.csv.gz'
    # fname = 'stat_petr.csv'
    path_to_incoming = '/home/guyos/Yandex.Disk/raxel/%s' % fname

    incoming_points = pr.read_points(path_to_incoming)
    # track = incoming_points.copy()

    grouped = incoming_points.groupby('IncomingTrackId')
    positions = []
    max_amp = 10
    bad_ids = []
    for name, track in grouped:
        track = ways.delete_outliers_from_track(track)
        if len(track) == 0 or track['AccelerationXOriginal'].sum() == 0:
            bad_ids.append(name)
            continue
        track.index = track.PointDate.values
        track = add_phone_cols(track)

        pos = stationary_positions(track, name, max_amp=max_amp, min_secs=5 * 60,
                                   max_residual_std=8)  # 1.3
        positions.append(pos)
        s = pos['time_delta'].sum()
        print(name, s)
    poss = pd.concat(positions)
    print(poss)
    # poss[poss.theta_0 > 130]
    # poss.to_csv('/home/guyos/Yandex.Disk/raxel/poss.csv')
    #
    # poss.plot(x='theta_0', y = 'rstd', kind='scatter')

    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()

    tracks = pr.read_points('/home/guyos/Yandex.Disk/raxel/0dbe23cb-db3d-4f3a-ada1-a700f1e9b668_tracks.csv.gz')
    m = poss.merge(tracks[['IncomingTrackId', 'TrackOrigin']], left_on='name', right_on='IncomingTrackId')
    m.to_csv('/home/guyos/Yandex.Disk/raxel/poss.csv')
    m['type'] = le.fit_transform(m['TrackOrigin'])

    data = poss[['time_delta', 'theta_0', 'rstd']]

    p = plt.scatter(x=m['rstd'], y=m['theta_0'], c=m['type'], s=m['time_delta'],
                    alpha=0.3, edgecolors='none')
    plt.xlabel('rstd')
    plt.ylabel('theta_0')
    plt.show()

    le.classes_

    ########################################

    name = 385504
    track = grouped.get_group(name)
    track = ways.delete_outliers_from_track(track)
    track.index = track.PointDate.values
    track = add_phone_cols(track)
    track['max_amplitude'] = max_amp

    track['acc'] = track['Speed'].diff().fillna(method='ffill').fillna(method='bfill')
    # pre_samp = track['acc'].resample('S').interpolate(method='linear')
    # resampled = track['acc'].rolling(window=10, center=True, win_type='parzen'). \
    #     mean().fillna(method='ffill').fillna(method='bfill')
    # track['smoothed_acc'] = resampled[track.index].values
    # track['smoothed_acc'] = track['smoothed_acc']/np.abs(np.max(track['smoothed_acc']))

    track['accZdiff'] = track['AccelerationZOriginal'].diff().fillna(method='ffill').fillna(method='bfill')
    # pre_samp = track['accZdiff'].resample('S').interpolate(method='linear')
    # resampled = track['accZdiff'].rolling(window=10, center=True, win_type='parzen'). \
    #     mean().fillna(method='ffill').fillna(method='bfill')
    # track['smoothed_accZdiff'] = resampled[track.index].values
    # track['smoothed_accZdiff'] = track['smoothed_accZdiff']/np.abs(np.max(track['smoothed_accZdiff']))

    track.plot(x='acc', y='AccelerationZOriginal',
               title=fname,
               kind='scatter', alpha=0.3)

    track.plot(x=track.index.get_level_values(0),
               title=fname,
               y=[
                   # 'Speed',
                   # 'acc',
                   # 'accZdiff',
                   # 'smoothed_acc',
                   # 'smoothed_accZdiff',
                   # 'smoothed_Az',
                   # 'angular_velocity',
                   'theta',
                   'smoothed_theta',
                   'theta_amplitude',
                   'max_amplitude',
                   # 'R',
                   # 'GyroscopeXOriginal',
                   # 'GyroscopeYOriginal',
                   # 'GyroscopeZOriginal'
                   # 'AccelerationXOriginal',
                   # 'AccelerationYOriginal',
                   # 'AccelerationZOriginal'
               ],
               # ylim=(0, 180)
               )

    # track.plot(x='AccelerationXOriginal', y='AccelerationYOriginal', kind='scatter')


    # plt.xlabel('theta')
    # plt.ylabel('distribution')
    # plt.title(_id)
    # plt.hist(incoming_points['theta'],
    #          normed=True,
    #          label='theta',
    #          bins=50)
    # plt.show()

    # theta_ewma = pre_samp.ewm(span=100, freq="S").mean()

    # theta.plot()
    # smoothed_theta.plot()
    # theta_ewma.plot()

    # d[name] / 60
    # len(d)

    # pd.DataFrame.from_dict(d, orient='index').to_csv('/home/guyos/Yandex.Disk/raxel/d2.csv')
    subtr = np.concatenate([poss['name'].unique(), bad_ids])
    not_valid = set(incoming_points['IncomingTrackId'].unique()).difference(subtr)

    for name in not_valid:  # poss['name'].unique():
        track = grouped.get_group(name)
        track = ways.delete_outliers_from_track(track)
        track.index = track.PointDate.values
        track = add_phone_cols(track)
        track['max_amplitude'] = max_amp

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.close('all')
        ax = track.plot(x=track.index.get_level_values(0),
                        title=track.IncomingTrackId.iloc[0],
                        y=[
                            # 'Speed',
                            # 'angular_velocity',
                            'theta',
                            'smoothed_theta',
                            'theta_amplitude',
                            'max_amplitude',
                            # 'R',
                            # 'GyroscopeXOriginal',
                            # 'GyroscopeYOriginal',
                            # 'GyroscopeZOriginal'
                            # 'AccelerationXOriginal',
                            # 'AccelerationYOriginal',
                            # 'AccelerationZOriginal'
                        ], ylim=(0, 180))
        fig = ax.get_figure()
        fig.savefig('/home/guyos/Yandex.Disk/raxel/alex/not valid2/%s.png' % name, bbox_inches='tight')

    #################################################################################

    h1 = pd.read_csv('/home/guyos/Downloads/masha/h1.txt', sep=' ')
    h2 = pd.read_csv('/home/guyos/Downloads/masha/h2.txt', sep=' ')
    trip = pd.concat([h1, h2])
    trip.rename(columns={'accX': 'AccelerationXOriginal', 'accY': 'AccelerationYOriginal',
                         'accZ': 'AccelerationZOriginal', 'dateLocal': 'PointDate',
                         'gyroX': 'GyroscopeXOriginal', 'gyroY': 'GyroscopeYOriginal',
                         'gyroZ': 'GyroscopeZOriginal'}, inplace=True)
    trip.PointDate = pd.to_datetime(trip.PointDate)
    trip.index = trip.PointDate.values
    accs = trip[['AccelerationXOriginal', 'AccelerationYOriginal', 'AccelerationZOriginal']]
    trip['R'] = np.linalg.norm(accs, axis=1)
    trip['theta'] = np.degrees(np.arccos(accs.AccelerationZOriginal / trip['R']))
    trip.loc[trip['theta'].isnull(), 'theta'] = 0

    resampled = trip['theta'].rolling(window=200, center=True, win_type='parzen'). \
        mean().fillna(method='ffill').fillna(method='bfill')
    trip['smoothed_theta'] = resampled[trip.index].values

    trip.plot(x='PointDate',
              y=[
                  # 'AccelerationXOriginal',
                  # 'AccelerationYOriginal',
                  # 'AccelerationZOriginal',
                  'theta',
                  'smoothed_theta'
              ])
