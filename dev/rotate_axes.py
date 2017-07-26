from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def posix_time(dt):
    '''
    Apply this function instead of to_datetime, if last doesn't work in your environment.
    '''
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)


def LPF(y, alpha, N):
    '''
    Apply low-pass filter to array y[:N].
    '''
    x = [y[0]]
    #    [x.append(y[i]) for i in range(1,N)] if you use this line, you will not apply LPF to data
    [x.append(y[i] * alpha[i] + (1 - alpha[i]) * x[-1]) for i in range(1, N)]
    return np.array(x)


def Angle(ivalue, y, z, ok, alpha, end):
    '''
    Calculate angle for rotation algorithm.
    '''
    x = [ivalue]
    [x.append(y[i] / z[i] * alpha[i] + x[-1] * (1 - alpha[i]) if ok[i] else x[-1]) for i in range(1, end)]
    return np.array(x)


def get_gyro_acc_z_rotated(init_track, date_column):
    '''
    Masha's & Arseniy's algorithm.

    Get track and name of date_column, e.g. PointDate or DateLocal.
    Apply low-pass filter to smooth data.
    Rotate axes to match z-axis with z-gravity.
    Recalculate AccZ and GyroZ.
    Return initial track with additional columns.

    All constants were chosen empirically.

    Main idea of algorithm: we rotate axes on each step using rotated values from previous step.
    Algorithm has two parts: rotation of initial part (200 first points) and rotation of remaining part. 

    Time complexity: O(N) where N is length of table (one cycle over the table).
    '''

    track = init_track.copy()
    Len = len(track)
    N = 200

    track[date_column] = pd.to_datetime(track[date_column])
    track[date_column] = track[date_column] + pd.Timedelta(hours=3)

    track['timestamp'] = pd.to_datetime(track[date_column]).apply(posix_time)
    delta_time = np.nan_to_num(track['timestamp'].diff())
    alphaLPF = np.nan_to_num(delta_time / (0.5 + delta_time))

    LPFAccX = LPF(track.AccelerationXOriginal.values, alphaLPF, Len)
    LPFAccY = LPF(track.AccelerationYOriginal.values, alphaLPF, Len)
    LPFAccZ = LPF(track.AccelerationZOriginal.values, alphaLPF, Len)
    LPFGyroX = LPF(track.GyroscopeXOriginal.values, alphaLPF, Len)
    LPFGyroY = LPF(track.GyroscopeYOriginal.values, alphaLPF, Len)
    LPFGyroZ = LPF(track.GyroscopeZOriginal.values, alphaLPF, Len)

    a_yz = (LPFAccY ** 2 + LPFAccZ ** 2) ** 0.5
    a_xyz = (LPFAccX ** 2 + LPFAccY ** 2 + LPFAccZ ** 2) ** 0.5
    gravity = (a_yz < 1.3) & (a_xyz > 0.7)
    alpha = np.nan_to_num(delta_time / (0.03 + delta_time))

    sinA = np.zeros(Len)
    cosA = np.zeros(Len)
    sinB = np.zeros(Len)
    cosB = np.zeros(Len)
    sinA[:N + 1] = Angle(0, LPFAccY, a_yz, gravity, alpha, N + 1)
    cosA[:N + 1] = Angle(1, LPFAccZ, a_yz, gravity, alpha, N + 1)
    sinB[:N + 1] = Angle(0, LPFAccX, a_xyz, gravity, alpha, N + 1)
    cosB[:N + 1] = Angle(1, a_yz, a_xyz, gravity, alpha, N + 1)

    AZ_ad = np.zeros(Len)
    AZ_ad[:N + 1] = (LPFAccZ[:N + 1] * cosA[:N + 1] + LPFAccY[:N + 1] * sinA[:N + 1]) * cosB[:N + 1] + LPFAccX[
                                                                                                       :N + 1] * sinB[
                                                                                                                 :N + 1]

    for i in range(N, len(sinA)):
        gravity[i] = (0.85 > AZ_ad[i - 1]) or (AZ_ad[i - 1] > 1.15)
        if gravity[i]:
            sinA[i] = alpha[i] * (LPFAccY[i] / a_yz[i]) + (1 - alpha[i]) * sinA[i - 1]
            cosA[i] = alpha[i] * (LPFAccZ[i] / a_yz[i]) + (1 - alpha[i]) * cosA[i - 1]
            sinB[i] = alpha[i] * (LPFAccX[i] / a_xyz[i]) + (1 - alpha[i]) * sinB[i - 1]
            cosB[i] = alpha[i] * (a_yz[i] / a_xyz[i]) + (1 - alpha[i]) * cosB[i - 1]
        else:
            sinA[i] = sinA[i - 1]
            cosA[i] = cosA[i - 1]
            sinB[i] = sinB[i - 1]
            cosB[i] = cosB[i - 1]
        AZ_ad[i] = (LPFAccZ[i] * cosA[i] + LPFAccY[i] * sinA[i]) * cosB[i] + LPFAccX[i] * sinB[i]

    track['GyroZ_adapted'] = (LPFGyroZ * cosA + LPFGyroY * sinA) * cosB + LPFGyroX * sinB
    track['AccZ_adapted'] = AZ_ad

    return track


def get_z_features(track):
    '''
    Input: track
    Output: 4 features: mean(gyro_z_small), std(gyro_z_small), mean(acc_z_small), std(acc_z_small)
    If track is small or there is no correct acceleration/gyroscope data - return 0 
    instead of corresponding statistics.
    '''
    res = np.zeros(4)
    if np.isnan(track.AccelerationXOriginal.values[0]) or np.sum(track.AccelerationXOriginal.values) == 0 \
            or len(track) < 300:
        return res

    track.sort_values(by='PointDate', inplace=True)
    track = get_gyro_acc_z_rotated(track, 'PointDate')

    acc_z = track['AccZ_adapted'].values
    gyro_z = track['GyroZ_adapted'].values

    acc_small = acc_z[np.abs(acc_z) < 1.]
    gyro_small = gyro_z[np.abs(gyro_z) < 0.1]

    if len(gyro_small) >= 300:
        res[0] = np.mean(gyro_small)
        res[1] = np.std(gyro_small)
    if len(acc_small) >= 300:
        res[2] = np.mean(acc_small)
        res[3] = np.std(acc_small)

    return res
