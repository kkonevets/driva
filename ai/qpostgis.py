#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:24:01 2016

Module for quiring Postgis SQL, common procedures

@author: guyos
"""

import numpy as np
import pandas as pd
import psycopg2
from shapely.ops import cascaded_union
from shapely.wkt import loads


def get_stops_in_geom(lpoly, host='localhost'):
    conn = psycopg2.connect(dbname='osm', host=host,
                            port=5432, user='postgres')
    cur = conn.cursor()

    good_stops = []
    stops_names = []

    query = '''
    SELECT
    ST_X(ST_Transform (way, 4326)) as Longitude,
       ST_Y(ST_Transform (way, 4326)) as Latitude,
      name, public_transport from public.planet_osm_point
    where highway='bus_stop' and
      ST_Intersects(way,ST_Transform(ST_GeomFromText('%s', 4326),900913));''' % lpoly.wkt

    cur.execute(query)
    res = cur.fetchall()

    for elem in res:
        good_stops.append((elem[0], elem[1]))
        stops_names.append(elem[2])

    good_stops = np.array(good_stops)
    return good_stops, stops_names


def get_projected_ways(mline, host='localhost'):
    """
    This method makes implicit projections of ways that intersect
    a buffered track onto a track itself, than calculates projected
    lengths and returns fractions grouped by way type.
    Main steps:
    1. read mline in temp table
    2. make polygon out of mline with buffer
    3. intersect the polygon with ways, then split these intersections
    in parts not exceeding some length (10 meters for example), setting it
    too small encure more precision, setting it too high encure high speed
    but low precision
    4. take points in splitted intersections, project them onto mline,
    connect projected points with lines, points are connected only for each
    intersection keeping the order
    5. resulted projected lines are wrapped with little buffer for
    convenience of union procedure. You cannot just make union of lines -
    they should not double
    6. calculate lengths of polygons for each way type devided by mline length
    :param mline: MultiLineString
    :param host:
    :return: fractions of ways in mline grouped by type of way
    """
    conn = psycopg2.connect(dbname='osm', host=host,
                            port=5432, user='postgres')

    cur = conn.cursor()

    query = '''
    SET work_mem TO '100MB';
    
    DROP TABLE IF EXISTS cashed_mline, projections;
    
    CREATE TEMP TABLE cashed_mline AS
    SELECT
        ST_GeomFromText('%s', 4326)
        AS mline;
    
    CREATE TEMP TABLE geoms AS
    SELECT
    -- 	ST_AsText(ST_Transform(
        ST_Transform((gdump).geom, 900913)
    -- 	,4326)) 
        as line,
        (gdump).path as path,
        ST_Transform(ST_Buffer((gdump).geom::geography, 50)::geometry, 900913) AS poly
    FROM (
        SELECT
            ST_Dump(mline) AS gdump
        FROM cashed_mline
    ) as dumped;
    
    
    CREATE TEMP TABLE raw_intersects AS
    SELECT
        geoms.path as path,
    -- -	ST_AsText(ST_Transform(
        ST_Intersection(ppol.way, geoms.poly)
    -- 	,4326))
        AS inter,
        ppol.railway as name
    FROM public.planet_osm_line AS ppol
    INNER JOIN geoms
    ON ppol.railway in ('rail', 'subway', 
        'tram', 'monorail', 'light_rail', 'narrow_gauge') 
        AND ST_Intersects(ppol.way, geoms.poly)
    UNION 
    SELECT
        geoms.path,    
        ST_Intersection(ppol.way, geoms.poly),
        ppol.route
    FROM public.planet_osm_line AS ppol
    INNER JOIN geoms
    ON ppol.route in ('bus', 'trolleybus', 'share_taxi') 
        AND ST_Intersects(ppol.way, geoms.poly)
    ;
    
    
    CREATE TEMP TABLE intersects AS
    SELECT 
        path,
        name,
        CASE 
            WHEN ST_GeometryType(inter) = 'ST_MultiLineString' THEN
                (ST_Dump(inter)).geom
            ELSE
                inter
        END AS inter
    FROM raw_intersects
    WHERE ST_GeometryType(inter) in ('ST_LineString', 'ST_MultiLineString');
    
    DROP TABLE raw_intersects;
    
    CREATE TEMP TABLE start_end AS
    SELECT 
        ints.path,
        name,
        ST_LineLocatePoint(line, ST_StartPoint(inter)) as p1,
        ST_LineLocatePoint(line, ST_EndPoint(inter)) as p2
    FROM intersects as ints
    INNER JOIN geoms
    ON geoms.path = ints.path;
    
    DROP TABLE intersects;
    
    CREATE TEMP TABLE sub_lines AS
    SELECT
        s_e.path,
        name,
        ST_LineSubstring(line, LEAST(p1, p2), GREATEST(p1, p2)) AS sub_line
    FROM start_end as s_e
    INNER JOIN geoms
    ON geoms.path = s_e.path;
    
    DROP TABLE start_end, geoms;
    
    CREATE TEMP TABLE projections AS
    SELECT
        path, 
        name,
        -- 	ST_AsText(ST_Transform(
        ST_Union(ST_Buffer(sub_line, 0.01)) 
        -- 	,4326)) 
        as proj
    FROM sub_lines
    GROUP BY path, name;
    
    DROP TABLE sub_lines;
    
    SELECT
        'track' as name,
        ST_Length2D_Spheroid(mline, 
            'SPHEROID["GRS 1980",6378137,298.257222101]')/1000 as length
    FROM cashed_mline
    UNION
    SELECT
        name,
        sum(ST_Length2D_Spheroid(ST_Transform(proj, 4326), 
        'SPHEROID["GRS 1980",6378137,298.257222101]'))/2000
    FROM projections
    group by name
    order by length desc
    ''' % mline.wkt

    try:
        cur.execute(query)
    except psycopg2.Error as err:
        conn.rollback()
        # with open("/var/www/uploads/errors.txt", "a") as error_file:
        #     error_file.write("got one\n")
        return pd.Series()

    res = cur.fetchall()
    res = pd.DataFrame(res, columns=['name', 'length'])
    res.set_index('name', inplace=True)
    if len(res) == 0:
        return pd.Series()
    return res['length']

#############################################################################

# import numpy as np
# import pandas as pd
# from ai.company_rdp import rdp
# import ai.tripmatching as tm
# import shapely.geometry as geom
# import psycopg2
# from shapely.wkt import loads
#
# from math import hypot
#
# import shapely.geometry as geom
# from sklearn.cluster import DBSCAN
#
# import ai.tripmatching as tm
# from ai.company_rdp import rdp
#
# from shapely.ops import cascaded_union
#
#
#
#
# track = pd.read_csv('/home/guyos/Yandex.Disk/company/dima.csv')
# dfl = delete_outliers_from_track(track)
# dfl.plot.scatter(x='Longitude', y='Latitude')
# np.unique(dfl['labels'])
# grouped = dfl.groupby('labels')
#
# lines = []
# for _, sub_track in grouped:
#     rdped = np.array(rdp(sub_track[['Longitude', 'Latitude']],
#                          epsilon=25 * tm.meter))
#     lines.append(rdped)
#
# mline = geom.MultiLineString(lines)
# conn = psycopg2.connect(dbname='osm', host='localhost',
#                         port=5432, user='postgres')
#
# cur = conn.cursor()
#
#
# query = '''
#     DROP TABLE IF EXISTS cashed_line, cashed_poly, rails, projections;
#
#     CREATE TEMP TABLE cashed_line AS
#     SELECT
#       ST_Transform(ST_GeomFromText('%s', 4326),900913)
#       AS mline;
#
#     CREATE TEMP TABLE cashed_poly AS
#     SELECT
#       ST_Buffer(cashed_line.mline, 50, 'endcap=flat join=round') AS mpoly
#     FROM cashed_line;
#
#     CREATE TEMP TABLE rails AS
#     SELECT
#     --     ST_AsText(ST_Transform(
#       ST_Segmentize((ST_Dump(ST_Intersection(ppol.way, cashed_poly.mpoly))).geom, 10)
#     --     ,4326))
#       AS inter,
#       ppol.railway
#     FROM public.planet_osm_line AS ppol
#     INNER JOIN cashed_poly
#     ON ppol.railway IS NOT NULL AND ST_Intersects(ppol.way, cashed_poly.mpoly);
#
#     CREATE TEMP TABLE projections AS
#     SELECT
#     --  ST_AsText(ST_Transform(
#       ST_MakeLine(ST_ClosestPoint(cashed_line.mline, dumped_points.point))
#     --    ,4326))
#       as proj,
#       MAX(dumped_points.railway) as railway
#     FROM (
#     SELECT
#       (ST_DumpPoints(rails.inter)).geom as point,
#       (ST_DumpPoints(rails.inter)).path as path,
#       rails.inter as inter,
#       rails.railway as railway
#     FROM rails
#     ORDER BY inter, path
#     ) as dumped_points
#     INNER JOIN cashed_line
#     ON TRUE
#     GROUP BY dumped_points.inter;
#
#     SELECT
#       ST_AsText(ST_Transform(proj, 4326)),
#       railway
#     FROM projections
#
#     --     SELECT
#     -- --       ST_AsText(ST_Transform(
#     --       ST_Length((ST_Dump(ST_Union(ARRAY(SELECT ST_MakeValid(proj) FROM projections)))).geom),
#     --       ST_Length(cashed_line.mline)
#     -- --       ,4326))
#     --     FROM cashed_line
#         ''' % mline.wkt
#
# try:
#     cur.execute(query)
# except psycopg2.Error as err:
#     conn.rollback()
#     print(err)
#
# res = cur.fetchall()
#
# linestr = [loads(elem[0]) for elem in res]
#
# from matplotlib import pyplot as plt
# import matplotlib.cm as cm
#
# for line in list(mline):
#     xy = line.xy
#     plt.plot(xy[0], xy[1], 'ro', c='green')
#
# colors = cm.rainbow(np.linspace(0, 1, len(linestr)))
# for l, c in zip(linestr, colors):
#     xy = l.xy
#     plt.plot(xy[0], xy[1], c=c)
#
# # track[['Latitude','Longitude']].to_csv('/home/guyos/Downloads/dell1.csv')
#
# fracs = get_projected_railways(mline)
# print(sum(fracs))
#
# # sum([l.length for l in linestr])/mline.length

# X.to_csv('/home/guyos/Downloads/X.csv')