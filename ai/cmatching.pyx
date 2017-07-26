# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:50:26 2016

@author: Kirill Konevets
email:   kkonevets@gmail.com

CONTENT:
    
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

def norm(point):
    """
    Euclidean distance of vector
    """
    return np.sqrt(np.power(point[0], 2) + np.power(point[1], 2))

def get_perp_intervals(dot1, dot2, threshold, rot_matrix):
    direction = dot2 - dot1
    perpendicular = np.dot(rot_matrix, direction)
    norm_p = norm(perpendicular)
    if norm_p == 0:
        return []
    unit_perp = perpendicular / norm_p
    moove = lambda x: np.array(
        [x + threshold * unit_perp, x - threshold * unit_perp])
    p1 = moove(dot1)
    p2 = moove(dot2)
    return [p1, p2]

def get_convex_hull(dots, threshold, left_side = True, right_side = True):
    if len(dots) <= 1:
        return dots
    hull = []
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    prev = dots[0]
    for dot in dots[1:]:
        perps = get_perp_intervals(prev, dot, threshold, rot_matrix)
        if len(perps) == 0:
            continue
        p1, p2 = perps

        if left_side and right_side:
            hull.append([p1[0], p2[0], p2[1], p1[1]])
        else:
            if left_side:
                hull.append([p1[0], p2[0], dot, prev])
            if right_side:
                hull.append([prev, dot, p2[1], p1[1]])

        prev = dot

    poligons = [Polygon(p) for p in hull]
    cas = cascaded_union(poligons)
    #    x, y = cas.exterior.coords.xy
    return cas
