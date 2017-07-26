import numpy as np
import ai.features as ft


def is_bicycle(track):
    """
        By Maria Klimchuck
    """
    if 'GyroscopeXOriginal' not in track.columns:
        return False

    track = track.copy()

    track.sort_values('PointDate', inplace=True)
    track.drop_duplicates(['Latitude', 'Longitude'], keep='last', inplace=True)
    track.drop_duplicates(['PointDate'], keep='last', inplace=True)
    track.reset_index(inplace=True)

    maxSpeed = track.Speed.max()
    count = ft.get_total_secs(track)

    track['gyr'] = np.linalg.norm(
        track[['GyroscopeXOriginal', 'GyroscopeYOriginal', 'GyroscopeZOriginal']], axis=1)
    track['gyr'].fillna(0, inplace=True)
    track2 = track.iloc[:-100].copy()
    if (len(track2) > 0) and ((len(track2[track2['gyr'] > 1]) / len(track2)) > 0.15) and (maxSpeed > 10) and (
                (count / 60) > 5) and (len(track) > 100) and (maxSpeed < 30):
        return True
    else:
        return False
