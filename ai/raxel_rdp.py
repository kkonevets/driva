"""
rdp
~~~

Python implementation of the Ramer-Douglas-Peucker algorithm.

"""

import numpy as np


def line_dists(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.cross(vec, start - points)
    return np.divide(abs(cross), np.linalg.norm(vec))


def rdp(M, epsilon=0):
    M = np.array(M)
    start, end = M[0], M[-1]
    dists = line_dists(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > epsilon:
        result1 = rdp(M[:index + 1], epsilon)
        result2 = rdp(M[index:], epsilon)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result
