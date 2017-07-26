#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 19:19:27 2016

@author: guyos
"""

import pickle

import gmplot
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

import ai.geoftrs as gf
import ai.ways as ways
from sklearn.model_selection import GridSearchCV
import streem.processing as proc

import batch.extract as bf

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import seaborn as sns
import itertools

import multiprocessing as mp
from functools import partial

sns.set(color_codes=True)

#####################################################################################


def delete_irrelevant_features(ftrs):
    irrel = np.array(['dist_from_last_track', 'end_count_', 'rel_end_count_',
                      'trips_per_day', 'trips_odd', 'rel_match_count_',
                      'height', 'day', 'minute', 'hour'])
    cols = [min(not (ir in c) for ir in irrel) for c in ftrs.columns]
    newftrs = ftrs[ftrs.columns[cols]]
    return newftrs


def extract_features_wrapper(path_to_tracks, path_to_rich, path_to_incoming,
                             use_features=None, host='localhost'):
    if use_features is None:
        use_features = []
    ftrs = bf.batch_extract_features(path_to_tracks, path_to_rich,
                                     path_to_incoming,
                                     use_geo_ftrs=True,
                                     host=host)
    ftrs = delete_irrelevant_features(ftrs)
    if len(use_features):
        ftrs = ftrs[use_features]

    return ftrs


def get_data(data_dir, host='localhost'):
    path_to_tracks = data_dir + 'data_tracks.csv.gz'
    path_to_rich = data_dir + 'data_rich.csv.gz'
    path_to_incoming = data_dir + 'data_incoming.csv.gz'
    ftrs = extract_features_wrapper(path_to_tracks, path_to_rich, path_to_incoming, host=host)

    tracks = pd.read_csv(path_to_tracks)
    merged = pd.merge(ftrs, tracks[['IncomingTrackId', 'TrackOrigin']], left_index=True,
                      right_on='IncomingTrackId')
    index = merged.IncomingTrackId
    X = merged.drop(['IncomingTrackId', 'TrackOrigin'], axis=1)
    X.index = index
    y = (merged['TrackOrigin'] == 'Bus').astype(int).values

    data = [X, y]
    with open(data_dir + 'ftrs.pickle2', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y


def select_features_from_model(model, X_train, y_train, X_test, y_test):
    """Fit model using each importance as a threshold"""
    thresholds = np.unique(model.feature_importances_)
    thresholds = np.sort(thresholds)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = get_model()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)

        accuracy, f1score, auc = predict(selection_model, select_X_test, y_test)
        print("""Thresh=%.3f, n=%d, Accuracy: %.2f%%, f1 score: %.2f%%, auc score: %.2f%%""" %
              (thresh, select_X_train.shape[1], accuracy, f1score, auc))

    selector = model.feature_importances_ > 0.021

    return selector

######################################################################################


def get_model():
    return xgb.XGBClassifier(colsample_bytree=0.8, gamma=5,
                             max_depth=6, n_estimators=300,
                             reg_alpha=2.5, subsample=0.9)
    # return xgb.XGBClassifier(max_depth=4, n_estimators=200,
    #                          learning_rate=0.1, min_child_weight=4,
    #                          colsample_bytree=0.8, reg_alpha=0, gamma=1, subsample=0.8)


def train_model(X, y, show_importance=False):
    model = xgb.XGBClassifier()
    cv_params = {
                'n_estimators': [1000],
                'learning_rate': [0.1],
                'max_depth': [5],
                'min_child_weight': [1],
                'gamma': [0],
                # 'reg_alpha': [2.25, 2.5, 2.75],
                # 'reg_lambda': [1,2,5,10]
    }
    keeper = xgb_grid_search(model, X, y, cv_params, nfold=5)
    res = pd.DataFrame(np.concatenate(keeper),
                       columns=['params', 'train_auc_mean', 'test_auc_mean'])
    res['diff'] = res['train_auc_mean'] - res['test_auc_mean']
    # can't overfitt more than 5 %
    sub = res[res['diff'] <= 0.05].copy()

    sub.sort_values(by=['test_auc_mean'], inplace=True,
                    ascending=[False])
    best = sub.iloc[0]
    best_params = best['params']
    print(best_params)
    print('test_auc_mean %f, train_auc_mean %f' %
          (best['test_auc_mean'], best['train_auc_mean']))

    sub.to_csv('/home/guyos/Downloads/scores.csv')

    if show_importance:
        # plot feature importance
        pyplot.close('all')
        xgb.plot_importance(model)
        pyplot.show()

    return model


def evaluate(probs, y_test, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1score = f1_score(y_test, y_pred) * 100
    if len(np.unique(y_test)) != 1:
        auc = roc_auc_score(y_test, probs) * 100
    else:
        auc = 0
    print("Accuracy: %.2f%%, f1 score: %.2f%%, auc score: %.2f%%" % (accuracy, f1score, auc))

    cm = confusion_matrix(y_test, y_pred)
    return cm


##############################################################################


def gplot_tracks(track, fname="mymap.html"):
    gmap = gmplot.GoogleMapPlotter(np.median(track.Latitude),
                                   np.median(track.Longitude), 12)

    start = track.iloc[0]
    end = track.iloc[-1]

    # gmap.plot(track.Latitude.values, track.Longitude.values, 'cornflowerblue', edge_width=10)
    gmap.scatter(track.Latitude.values, track.Longitude.values,
                 '#3B0B39', size=5, marker=False)
    # gmap.scatter([start.Latitude], [start.Longitude], 'g', marker=True)
    # gmap.scatter([end.Latitude], [end.Longitude], 'y', marker=True)

    gmap.draw(fname)


def bus_inspection(data_dir, host='localhost'):
    path_to_incoming = data_dir + 'data_incoming.csv.gz'
    path_to_tracks = data_dir + 'data_tracks.csv.gz'

    tracks = pd.read_csv(path_to_tracks)
    tracks = tracks[tracks['TrackOrigin'] == 'Bus']
    incoming_points = proc.read_points(path_to_incoming)
    incoming_points = incoming_points[incoming_points['IncomingTrackId'].isin(tracks['IncomingTrackId'])]
    inspection_plots(incoming_points, host='localhost')


def inspection_plots(incoming_points, host='localhost'):
    grouped = incoming_points.groupby('IncomingTrackId')
    for _id, track in grouped:
        print(_id)
        gplot_tracks(track, fname=data_dir + 'plots/' + "%s.html" % _id)

        track = ways.delete_outliers_from_track(track)
        if len(track) > 0:
            bus_stops, names = gf.get_bus_stops(track, host)
            fig = gf.plot_bus_stops(track, bus_stops)
            fig.savefig(data_dir + '/plots/%s.png' % _id)


def unite_users(data_dir, host='localhost'):
    from test.test import zipped_tripples
    zipped_users = zipped_tripples(data_dir, n1='tracks', n2='rich', n3='incoming')
    incoming_points = pd.concat([proc.read_points(c) for a,b,c in zipped_users])
    tracks = pd.concat([proc.read_points(a) for a,b,c in zipped_users])
    incoming_points = incoming_points[incoming_points['IncomingTrackId'].isin(tracks['IncomingTrackId'])]

    inspection_plots(incoming_points, host=host)


##############################################################################


if __name__ == '__main__':
    data_dir = '/home/guyos/Documents/bus_detection/50_50/'
    # data_dir = '/home/guyos/Yandex.Disk/company/data/dima/'
    host = 'localhost'  # 'zeus.local'

    # bus_inspection(data_dir)

    #################################
    bus_ids = tuple(pd.read_csv('./etc/real_busses.csv').Id.values)
    tracks = pd.read_csv(data_dir + 'data_tracks.csv.gz')
    tracks.loc[tracks['Id'].isin(bus_ids), 'TrackOrigin'] = 'Bus'
    tracks.loc[~tracks['Id'].isin(bus_ids), 'TrackOrigin'] = 'OriginalDriver'
    tracks.to_csv(data_dir + 'data_tracks.csv.gz', compression='gzip',
              index=False, encoding='utf8')
    #################################

    X, y = get_data(data_dir, host=host)
    with open(data_dir + 'ftrs.pickle', 'rb') as handle:
        X, y = pickle.load(handle)

    X.to_csv(data_dir+'X.csv')
    pd.DataFrame(y).to_csv(data_dir+'y.csv', index=False)
    y = pd.read_csv(data_dir+'y.csv')

    dtrain = xgb.DMatrix(X, y)
    model = get_model()
    res = xgb.cv(model.get_params(), dtrain, nfold=5, num_boost_round=382,
                 metrics=['auc'], seed=0, stratified=True, as_pandas=True)
    print(res['test-auc-mean'].iloc[-1])
    print(res['train-auc-mean'].iloc[-1])
    # 0.89310085



    params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'gamma': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 2.5,
            'subsample': 0.9
    }

    estop = 40
    dtrain = xgb.DMatrix(X, y)
    res = xgb.cv(model.get_params(), dtrain, metrics=['auc'],
                 num_boost_round=1000000000, nfold=10,
                 early_stopping_rounds=estop, seed=0, stratified=True)

    best_nrounds = res.shape[0] - estop
    best_nrounds = int(best_nrounds / 0.8)


    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    results = cross_val_score(model, X, y, cv=kfold, verbose=2, scoring='roc_auc', n_jobs=-1)  # roc_auc
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # Standardized: 94.18% (1.51%)

    #########################################################################################

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=223, stratify=y)

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    # y_pred = (probs >= 0.5).astype(int)
    # wrong_busses = X_test[(y_test == 1) & (y_pred == 0)].index
    # pd.Series(wrong_busses).to_csv(data_dir + 'wrong_busses.csv', index=False)

    ones = probs[np.where(y_test == 1)]
    zeros = probs[np.where(y_test == 0)]

    sns.distplot(ones, label='bus')
    sns.distplot(zeros, label='original driver')
    plt.xlabel('prob')
    plt.ylabel('freq')
    plt.legend(loc="best")

    cm = evaluate(probs, y_test, threshold=0.75)
    print(cm)
    print(cm[0,0]/sum(cm[0,]))
    print(cm[1,1]/sum(cm[1,]))

    threshold = .5
    while threshold < 1:
        cm = evaluate(probs, y_test, threshold=threshold)
        print(cm)
        print('threshold %s' % threshold)
        threshold += .05

    # df = pd.DataFrame()
    # df['IncomingTrackId'] = list(X_test.index)
    # df['prediction'] = ['Bus' if v==1 else 'OriginalDriver' for v in y_pred]
    # df['y_true'] = ['Bus' if v==1 else 'OriginalDriver' for v in y_test]
    # df.to_csv('~/pred.csv')


    #########################################################################################

    model.fit(X, y)

    with open(data_dir + 'model_busses.pickle', 'wb') as handle:
        pickle.dump([list(X.columns), model], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # #######################################################################################
    #
    # # data_dir = '/home/guyos/Yandex.Disk/company/'
    # # host = 'localhost'  # 'zeus.local'
    # #
    # # path_to_tracks = data_dir + 'RichTrack_dima_bus.csv'
    # # path_to_rich = data_dir + 'RichTrackPoints_dima_bus.csv'
    # # path_to_incoming = data_dir + 'IncomingTrackPoints_dima_bus2.csv'
    # # bulk_bus_ftrs = extract_features_wrapper(path_to_tracks, path_to_rich,
    # #                                          path_to_incoming,
    # #                                          host=host)
    # # bulk_bus_ftrs.to_csv(data_dir + 'ftrs.csv')
    #
    # # incoming_points = proc.read_points(path_to_incoming)
    # # track = rails.delete_outliers_from_track(incoming_points)
    # #
    # # bus_stops, names = gf.get_bus_stops(track, host)
    # # gf.plot_bus_stops(track, bus_stops)
    # #
    # # with open('./data/models/model_busses.pickle', 'rb') as handle:
    # #     columns, model = pickle.load(handle)
    # #
    # # model.predict(bulk_bus_ftrs[columns])
    # #
    # # from xgboost import plot_tree
    # #
    # # plot_tree(model, num_trees=5)
    # #
    # #######################################################################################
    #
    # data_dir = '/home/guyos/Documents/bus_detection/50_50/'
    # host = 'localhost'
    #
    # import ai.geoftrs as gf
    # import ai.rails as rails
    # import streem.processing as proc
    #
    # incoming_points = proc.read_points(data_dir + 'data_incoming.csv.gz')
    # tracks = proc.read_points(data_dir + 'data_tracks.csv.gz')
    # # tracks = tracks[tracks['TrackOrigin'] == 'Bus']
    # # tracks = tracks.loc[:20]
    # incoming_points = incoming_points[incoming_points.IncomingTrackId.isin(wrong_busses)]
    #
    # grouped = incoming_points.groupby('IncomingTrackId')
    #
    # for name, track in grouped:
    #     track = rails.delete_outliers_from_track(track)
    #
    #     bus_stops, names = gf.get_bus_stops(track, host)
    #     gf.plot_bus_stops(track, bus_stops, close_all=False)
    #
    #     # X.to_csv('/home/guyos/Downloads/X.csv')
    #     #
    #     # sub = X[X['TrackOrigin'] == 'Bus']
    #     # sub['ped_last_stop']

df = pd.read_csv('/home/guyos/Downloads/test.csv')
bus = df[df['Prediction'] == 'Bus']
orig = df[df['Prediction'] == 'OriginalDriver']

sns.distplot(bus.Distance, label='bus')
sns.distplot(orig.Distance, label='original')
plt.legend(loc="best")


path_to_tracks = data_dir + 'data_tracks.csv.gz'
tracks = pd.read_csv(path_to_tracks)
# tracks = tracks[tracks.Distance < 50]
bus = tracks[tracks['TrackOrigin'] == 'Bus']
orig = tracks[tracks['TrackOrigin'] == 'OriginalDriver']

sns.distplot(bus.MaxSpeed, label='bus')
sns.distplot(orig.MaxSpeed, label='original')
plt.legend(loc="best")


v = X.speed_max
bus_v = v[y==1]
orig_v = v[y==0]
sns.distplot(bus_v, label='bus')
sns.distplot(orig_v, label='original')
plt.legend(loc="best")

a = X.acc_mean
bus_a = a[y==1]
orig_a = a[y==0]
sns.distplot(bus_a, label='bus')
sns.distplot(orig_a, label='original')
plt.legend(loc="best")


_X = X[X.total_secs<3600]
_y = y[np.where(X.total_secs<3600)]
t = _X.total_secs
bus_t = t[_y==1]
orig_t = t[_y==0]
sns.distplot(bus_t, label='bus')
sns.distplot(orig_t, label='original')
plt.legend(loc="best")


r = X.Rating
bus_r = r[y==1]
orig_r = r[y==0]
sns.distplot(bus_r, label='bus')
sns.distplot(orig_r, label='original')
plt.legend(loc="best")


d = X.Distance.values
d = d[np.where(d<30)]
y = y[np.where(d<30)]
bus_d = d[np.where(y==1)]
orig_d = d[np.where(y==0)]
sns.distplot(bus_d, label='bus')
sns.distplot(orig_d, label='original')
plt.legend(loc="best")


from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

model = get_model()
model.fit(X, y)
# plot feature importance
plot_importance(model, max_num_features=20)
pyplot.tight_layout()
pyplot.show()

print(model.feature_importances_)
print(X.columns)
imp = pd.DataFrame()
imp['feature'] = list(X.columns)
imp['gain'] = model.feature_importances_
imp.sort_values(by='gain', ascending=False, inplace=True)
imp['feature'][:20]

from xgboost import plot_tree
plot_tree(model)

data = X.copy()
data['y'] = y
g = sns.pairplot(data, x_vars='y', hue='y')


from dev.rank import rank, plot_distributions

ranked = rank(X, y)
plot_distributions(X, y, ranked, nrows=5)
ranked.to_csv('/home/guyos/Downloads/ranked.csv')

m = pd.read_csv('/home/guyos/Downloads/masha.csv', usecols=['RichTrackId', 'Real Origin'])
tracks = pd.read_csv(data_dir+'tracks former/'+'data_tracks.csv.gz')

for n, row in m.iterrows():
    cur_id = row['RichTrackId']
    if row['Real Origin'] == 'Not bus':
        tracks.loc[tracks['Id'] == cur_id, 'TrackOrigin'] = 'OriginalDriver'
    elif 'Bus' == row['Real Origin']:
        tracks.loc[tracks['Id'] == cur_id, 'TrackOrigin'] = 'Bus'

tracks.to_csv(data_dir+'data_tracks.csv.gz', compression='gzip', index=False)

from collections import Counter

tracks = pd.read_csv(data_dir+'data_tracks.csv.gz')
c = Counter(tracks.TrackOrigin)
c['OriginalDriver']/(c['Bus']+c['OriginalDriver'])


sub = X[['rail_frequency', 'bus_frac', 'bus_uncovered_len', 'rail_uncovered_len']]

sub[sub['bus_frac'] > 2]

frac = X.bus_frac
bus_f = frac[y==1]
orig_f = frac[y==0]
sns.distplot(bus_f, label='bus')
sns.distplot(orig_f, label='original')
plt.legend(loc="best")


ul = X.bus_uncovered_len.values*1000
ul = ul[np.where(ul<10000)]
_y = y[np.where(ul<10000)]
bus_u = ul[_y==1]
orig_u = ul[_y==0]
sns.distplot(bus_u, label='bus')
sns.distplot(orig_u, label='original')
plt.legend(loc="best")

##
track = read_points('/home/guyos/Desktop/incPoints.csv')







