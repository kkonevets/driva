# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:09:27 2016

@author: Kirill Konevets
email:   kkonevets@gmail.com

CONTENT:
    
"""

from functools import partial

import ai.cmatching as cmatching
import pandas as pd
from ai.raxel_rdp import rdp

# SMOOTHING

# TODO: meter is not always this big, use haversine instead
meter = 8.9883063456985468e-06


def apply_rdp(points, epsilon=50, by='TrackId'):
    """
    Apply Ramer Douglas Peucker algorithm to each track in 'points'
    """
    grouped = points.groupby([by, 'StartDate'])
    smoothed = pd.DataFrame()
    for pair in grouped:
        (Id, StartDate), track = pair
        part = rdp(track[['Longitude', 'Latitude']], epsilon=epsilon * meter)
        part = pd.DataFrame(part, columns=['Longitude', 'Latitude'])
        part.insert(0, 'StartDate', StartDate)
        part.insert(0, by, Id)
        smoothed = smoothed.append(part)
    return smoothed


def get_coords(track):
    return track[['Longitude', 'Latitude']].as_matrix()


def get_polygons(points, threshold=30, by='TrackId',
                 left_side=True, right_side=True):
    grouped = points.groupby([by])
    wrapper = partial(cmatching.get_convex_hull,
                      threshold=threshold * meter,
                      left_side=left_side, right_side=right_side)
    polygons = grouped.apply(get_coords).apply(wrapper)

    return polygons
