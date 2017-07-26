#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:37:08 2016

@author: Kirill Konevets
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = '~/Downloads/'

fname = 'dima_acc.csv'

df = pd.read_csv(data_dir + fname, index_col='time', sep=';', decimal=',')
# df = pd.read_csv(data_dir+fname, index_col='loggingTime')
# df.index = pd.to_datetime(df.index)
# df.index = (df.index - min(df.index))/np.timedelta64(1, 's')

df['norm'] = np.linalg.norm(df[['X_value', 'Y_value', 'Z_value']], axis=1)
df['norm'] = df['norm'] / 9.81523
# df['norm'] = np.linalg.norm(df[['accelerometerAccelerationX',
#                                'accelerometerAccelerationY',
#                                'accelerometerAccelerationZ']], axis=1)
df['norm'] = df['norm'] - np.mean(df['norm'])

df2 = df[((df.index >= 30) & (df.index < 40))]
df2.index = df2.index - min(df2.index)

df = df2

N = len(df)
length = max(df.index)
# t = np.linspace(0,length, N)
t = df.index

Y = np.fft.fft(df.norm) / N
Y = np.fft.fftshift(Y)

freq = np.fft.fftfreq(N, length / (N - 1))
freq = np.fft.fftshift(freq)

pos_idxs = np.where(freq >= 0)
freq = freq[pos_idxs]
Y = Y[pos_idxs]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, df.norm)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('A[g]')
ax[1].plot(freq, abs(Y), 'r')  # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y|[g]')

# f - 1.5 - 2.5
# A - 0.09 - 2.3


df2.to_csv('/home/guyos/raxel/df2.csv')

# %%

N = 1000
length = 10
d = length / (N - 1)

t = np.linspace(0, length, N)
y = np.sin(2 * np.pi * t)

Fk = np.fft.fft(y) / N
Fk = np.fft.fftshift(Fk)

freq = np.fft.fftfreq(N, d)
freq = np.fft.fftshift(freq)

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(freq, abs(Fk), 'r')  # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

df.plot(x=df.index, y='norm')


#############################################################################################

data_dir = '~/Downloads/'
fname = 'dima_acc.csv'

df = pd.read_csv(data_dir + fname, index_col='time', sep=';', decimal=',')
df['norm'] = np.linalg.norm(df[['X_value', 'Y_value', 'Z_value']], axis=1)
df['norm'] = df['norm'] / 9.81523
df['norm'] = df['norm'] - np.mean(df['norm'])












res = pd.read_csv('~/Downloads/Stats10Or.csv')
res.plot.scatter(x='Freq', y='Ampl', alpha=0.3)
res['Ampl']
