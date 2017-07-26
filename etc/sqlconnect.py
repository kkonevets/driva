import pymssql
import pandas as pd
import numpy as np

with open('./etc/queries/tracks', 'r') as fquery:
    qtracks = fquery.read()
with open('./etc/queries/richpoints', 'r') as fquery:
    qrich = fquery.read()
with open('./etc/queries/incoming', 'r') as fquery:
    qinc = fquery.read()

# SELECT DeviceToken, sum(1) as num
#  FROM [MobileServiceStage].[dbo].[RichTracks] (nolock)
#  where [OriginChanged]=1 and TrackOrigin='Taxi'
#  group by DeviceToken
#  having sum(1) > 1
#  order by num desc


# server='company-dwhtest.companytelematics.local'
server = ''
user = ''
pas = ''
conn = pymssql.connect(server=server, user=user, password=pas)

data_dir = 'E:\\Kirill\\YandexDisk\\company\\data\\'


# users = ('f1de1c4b-323c-420a-a68f-ab8aa9960316',)
# users_series = pd.read_csv('./taxi_tokens.txt', header=None)
# users = np.squeeze(users_series.values)
#
# for user in users:
#    tracks = pd.read_sql(qtracks, conn,  params={"user":user})
#    tracks.to_csv(data_dir+"%s_tracks.csv.gz"%user,
#                  compression = 'gzip', index=False, encoding='utf8')
#    print('%s %s' % (user, 'tracks'))
#
#    rich = pd.read_sql(qrich, conn, params={"user":user})
#    rich.to_csv(data_dir+"%s_rich.csv.gz"%user,
#                compression = 'gzip', index=False)
#    print('%s %s' % (user, 'rich'))
#
#    inc = pd.read_sql(qinc, conn, params={"user":user})
#    inc.to_csv(data_dir+"%s_incoming.csv.gz"%user,
#               compression = 'gzip', index=False)
#    print('%s %s' % (user, 'inc'))


def get_gyro():
    q = """SELECT top 1000 *
    INTO #tracks
    FROM [MobileServiceStage].[dbo].[RichTracks] (nolock)
    where [TrackOrigin] = 'OriginalDriver'
    order by startdate desc

    SELECT inc.*
    from [MobileServiceIncoming].[dbo].[v_AllTrackPoints] as inc (nolock)
    inner join #tracks
    on #tracks.IncomingTrackId = inc.IncomingTrackId
    	and inc.[AccelerationXOriginal] is not null
     """
    l = []
    inc = pd.read_sql(q, conn, chunksize=400000)
    for i, df in enumerate(inc):
        l.append(df)
        print(i)

    inc = pd.concat(l)
    inc.to_csv(data_dir + 'data_incoming.csv.gz', compression='gzip',
               index=False, encoding='utf8')
    del inc


def chunck_query(connection, query_text, file_name, chunksize=200000):
    l = []
    reader = pd.read_sql(query_text, connection, chunksize=chunksize)
    for i, df in enumerate(reader):
        l.append(df)
        print(i)

    df = pd.concat(l)
    df.to_csv(file_name, compression='gzip', index=False, encoding='utf8')


def get_bus_data():
    qtemp = """SELECT *
    INTO #tracks
    FROM (SELECT top 537 *
     FROM [MobileServiceStage].[dbo].[RichTracks] (nolock)
      where [TrackOrigin] = 'Bus' and [OriginChanged] = 1 
      and Country in ('Россия','Malaysia','Singapore')
      and StartDate < '2017-05-13' and DistanceGps>2 
    order by Id desc
    ) as busses
    union 
    SELECT *
    FROM (SELECT top 537 *
    	FROM [MobileServiceStage].[dbo].[RichTracks] (nolock)
    	where [TrackOrigin] = 'OriginalDriver' and OriginChanged=1
    	and Country in ('Россия','Malaysia','Singapore')
       and StartDate < '2017-05-13' and DistanceGps>2 
    order by Id desc
    ) as cars

    """
    q1 = """
    SELECT *
    FROM #tracks
    """
    q2 = """
    SELECT rich.*
    FROM [MobileServiceStage].[dbo].[RichTrackPoints] as rich (nolock)
    inner join #tracks
    on rich.DeviceToken = #tracks.DeviceToken and rich.RichTrackId = #tracks.Id
    """
    q3 = """
    SELECT inc.*
    FROM [MobileServiceIncoming].[dbo].[v_AllTrackPoints] as inc (nolock)
    inner join #tracks
    on inc.DeviceToken = #tracks.DeviceToken and 
        inc.IncomingTrackId = #tracks.IncomingTrackId
    """

    tracks = pd.read_sql(qtemp + q1, conn)

#    bus_ids = tuple(pd.read_csv('./etc/real_busses.csv').Id.values)
    org_ids = tuple(pd.read_csv('./etc/originals.csv').Id.values)
#    tracks.loc[tracks['Id'].isin(bus_ids), 'TrackOrigin'] = 'Bus'
    tracks.loc[tracks['Id'].isin(org_ids), 'TrackOrigin'] = 'OriginalDriver'
    tracks.to_csv(data_dir + "data_tracks.csv.gz",
                  compression='gzip', index=False, encoding='utf8')


    chunck_query(conn, q2, data_dir + "data_rich.csv.gz", chunksize=200000)
    chunck_query(conn, q3, data_dir + "data_incoming.csv.gz", chunksize=200000)

