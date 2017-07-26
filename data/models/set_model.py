#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:01:06 2016

@author: Kirill Konevets
"""

import pickle

import redis

r = redis.Redis(host='localhost')

model = pickle.load(open("./data/models/model_busses.pickle", "rb"))
r.set('model_bus_detection', pickle.dumps(model))
