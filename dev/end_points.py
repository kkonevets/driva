
from datetime import datetime
from functools import partial
from itertools import cycle

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from haversine import haversine
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances


def get_stat(cluster):
    stat = {}
    stat['holy_num'] = sum(cluster.weekday.isin([6, 7]))
    stat['work_num'] = len(cluster) - stat['holy_num']
    stat['median_hour'] = np.median([p.time().hour for p in cluster.PointDate])
    stat['Longitude'] = cluster.Longitude.median()
    stat['Latitude'] = cluster.Latitude.median()

    return pd.DataFrame.from_dict(stat, orient='index').T


def annotate(clusters):
    clusters = clusters.copy()
    clusters.loc[:, 'place'] = ''

    # 1. detect holydays
    holy_idxs = clusters['holy_num'] > clusters['work_num']
    clusters.loc[holy_idxs, 'place'] = 'holy'

    # 2. detect home
    home_idx = np.argmax(clusters['holy_num'] + clusters['work_num'])
    home = clusters.loc[home_idx]
    clusters.loc[home_idx, 'place'] = 'home'

    # 3. detect work
    cond = ~clusters.index.isin([home_idx])
    all_but_home = clusters[cond]
    if len(all_but_home):
        work_idx = np.argmax(all_but_home['work_num'])
        work = clusters.loc[work_idx]
        clusters.loc[work_idx, 'place'] = 'work'

    # if home.median_hour >= 17 and home.median_hour <= 23:
    #        clusters.loc[home_idx, 'place'] = 'home'
    #    elif len(all_but_home) and work.median_hour >= 7 and work.median_hour <= 11:
    #        clusters.loc[work_idx, 'place'] = 'work'

    return clusters


def get_centers_worker(user_ends, rad=1):  # rad = 1 km
    points = user_ends.copy()
    X = points[['Latitude', 'Longitude']].values

    # need to calc bandwidth withot outliers
    dist_matrix = pairwise_distances(X, X, metric=haversine)
    neighbour_matrix = dist_matrix < rad
    neibs = [sum(row) - 1 >= 3 for row in neighbour_matrix]

    neighb_points = X[np.where(neibs)]
    valid_index = points.iloc[neibs].index

    if len(neighb_points):
        db = DBSCAN(eps=rad, min_samples=4, metric=haversine).fit(neighb_points)
    else:
        db = DBSCAN(eps=rad, min_samples=4, metric=haversine).fit(X)

    points.loc[valid_index, 'label'] = db.labels_
    points.loc[~points.index.isin(valid_index), 'label'] = -1

    labels = points['label'].astype(np.int).values
    valid_ends = points[points['label'] != -1]
    grouped_ends = valid_ends.groupby('label')
    clusters = grouped_ends.apply(get_stat)
    if len(clusters):
        clusters.index = clusters.index.droplevel(1)
        clusters = annotate(clusters)

    return clusters, labels


def get_centers(user_ends, rad=1):
    try:
        clusters, _ = get_centers_worker(user_ends, rad)
    except:
        print(set(user_ends.DeviceToken))
        return

    return clusters


if __name__ == '__main__':
    end_points = pd.read_csv('/home/guyos/Yandex.Disk/company/data/dots.csv')
    end_points.PointDate = end_points.PointDate.apply(dateutil.parser.parse)
    end_points['weekday'] = end_points.PointDate.apply(datetime.weekday) + 1

    end_points['Latitude'] = end_points['FinishLat']
    end_points['Longitude'] = end_points['FinishLon']

    grouped = end_points.groupby('DeviceToken')

    wrapper = partial(get_centers, rad=1)
    centers = grouped.apply(wrapper)

    centers.to_csv('/home/guyos/Downloads/dots_res2.csv')


    taxi_pass = pd.read_csv('/home/guyos/Yandex.Disk/company/data/taxi_pass.csv.gz')





