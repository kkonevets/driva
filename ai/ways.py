import numpy as np
import shapely.geometry as geom
from pyproj import Proj, transform
from sklearn.cluster import DBSCAN

import ai.qpostgis as qpg
import ai.tripmatching as tm
from ai.company_rdp import rdp


def delete_outliers_from_track(track):
    """Apply DBSCAN to track points to find clusters"""

    track = track.copy()

    track.sort_values('PointDate', inplace=True)
    track.drop_duplicates(['Latitude', 'Longitude'], keep='last', inplace=True)
    track.drop_duplicates(['PointDate'], keep='last', inplace=True)
    track.reset_index(inplace=True)

    if len(track) == 0:
        track['labels'] = None
        return track

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:3857')

    coords = track[['Longitude', 'Latitude']].values
    trans_coords = np.array(transform(inProj, outProj, coords[:, 0], coords[:, 1])).T
    data = trans_coords

    # db = DBSCAN(eps=0.1, min_samples=3, metric=haversine).fit(data)
    db = DBSCAN(eps=100, min_samples=3).fit(data)
    track['labels'] = db.labels_
    track = track[db.labels_ != -1]
    track = track.groupby('labels').filter(lambda x: len(x) > 4)

    return track


def way_features(track, host='localhost'):
    # dfl = delete_outliers_from_track(track)
    grouped = track.groupby('labels')

    lines = []
    for _, sub_track in grouped:
        rdped = np.array(rdp(sub_track[['Longitude', 'Latitude']],
                             epsilon=25 * tm.meter))
        lines.append(rdped)

    mline = geom.MultiLineString(lines)

    d = {}

    lengths = qpg.get_projected_ways(mline, host=host)
    if len(lengths):
        track_length = lengths['track']
    else:
        track_length = mline.length / tm.meter

    fracs = lengths / track_length

    rail_fracs = fracs[fracs.index.isin(('rail', 'subway',
                                         'tram', 'monorail',
                                         'light_rail', 'narrow_gauge'))]
    bus_fracs = fracs[fracs.index.isin(('bus', 'trolleybus',
                                        'share_taxi'))]
    d['rail_frequency'] = 0 if len(rail_fracs) == 0 else max(rail_fracs)
    d['bus_frac'] = 0 if len(bus_fracs) == 0 else max(bus_fracs)
    d['bus_uncovered_len'] = (1 - d['bus_frac']) * track_length
    d['rail_uncovered_len'] = (1 - d['rail_frequency']) * track_length

    return d


# track = pd.read_csv('/home/guyos/Yandex.Disk/track.csv')
# track = delete_outliers_from_track(track)
