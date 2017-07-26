import pymssql
import pandas as pd
import numpy as np


server = ''
user = ''
pas = ''
conn = pymssql.connect(server=server, user=user, password=pas)

data_dir = 'E:\\Kirill\\YandexDisk\\raxel\\data\\'


def chunck_query(connection, query_text, file_name, chunksize=200000):
    l = []
    reader = pd.read_sql(query_text, connection, chunksize=chunksize)
    for i, df in enumerate(reader):
        l.append(df)
        print(i)

    df = pd.concat(l)
    df.to_csv(file_name, compression='gzip', index=False, encoding='utf8')


bus_ids = tuple(pd.read_csv('./etc/real_busses.csv').Id.values)
org_ids = tuple(pd.read_csv('./etc/originals.csv').Id.values)

qtemp = '''
    select *
    INTO #bus_tracks
    from [MobileServiceStage].[dbo].[RichTracks] (nolock)
    where Id in %s
    
    select *
    INTO #org_tracks
    from [MobileServiceStage].[dbo].[RichTracks] (nolock)
    where Id in %s
    UNION
    select top 191 *
    from [MobileServiceStage].[dbo].[RichTracks] (nolock)
    where [TrackOrigin] = 'OriginalDriver' and OriginChanged=1
      and Country in ('Россия','Malaysia','Singapore') and CompanyId=6
      and DistanceGPS > 2
      and StartDate < '2017-05-12' and Id not in %s
    order by Id desc    
    ''' % (str(bus_ids), str(org_ids), str(bus_ids+org_ids))

q1 = """
SELECT *
FROM #bus_tracks
UNION
SELECT *
FROM #org_tracks
"""
q2 = """
SELECT rich.*
FROM [MobileServiceStage].[dbo].[RichTrackPoints] as rich (nolock)
inner join #bus_tracks
on rich.DeviceToken = #bus_tracks.DeviceToken and rich.RichTrackId = #bus_tracks.Id
UNION
SELECT rich.*
FROM [MobileServiceStage].[dbo].[RichTrackPoints] as rich (nolock)
inner join #org_tracks
on rich.DeviceToken = #org_tracks.DeviceToken and rich.RichTrackId = #org_tracks.Id

"""
q3 = """
SELECT inc.*
FROM [MobileServiceIncoming].[dbo].[v_AllTrackPoints] as inc (nolock)
inner join #bus_tracks
on inc.DeviceToken = #bus_tracks.DeviceToken and 
    inc.IncomingTrackId = #bus_tracks.IncomingTrackId
UNION
SELECT inc.*
FROM [MobileServiceIncoming].[dbo].[v_AllTrackPoints] as inc (nolock)
inner join #org_tracks
on inc.DeviceToken = #org_tracks.DeviceToken and 
    inc.IncomingTrackId = #org_tracks.IncomingTrackId

"""

tracks = pd.read_sql(qtemp + q1, conn)
tracks.loc[tracks['Id'].isin(bus_ids), 'TrackOrigin'] = 'Bus'
tracks.loc[~tracks['Id'].isin(bus_ids), 'TrackOrigin'] = 'OriginalDriver'

tracks.to_csv(data_dir + "data_tracks.csv.gz",
              compression='gzip', index=False, encoding='utf8')

chunck_query(conn, q2, data_dir + "data_rich.csv.gz", chunksize=200000)
chunck_query(conn, q3, data_dir + "data_incoming.csv.gz", chunksize=200000)
