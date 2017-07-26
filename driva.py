# -*- coding: utf-8 -*-]

from __future__ import print_function

import getopt
import os
import sys

import redis
from flask import Flask, request, render_template
from rq import Queue
from werkzeug import secure_filename

from streem.processing import Features, save_features

UPLOAD_FOLDER = '/var/www/uploads'
ALLOWED_EXTENSIONS = {'zip', 'gz', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'mobidick'

host = 'localhost'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def process_files(files):
    error = check_files(files)
    if error:
        return None, error
    paths = {}
    for key in files:
        f = files[key]
        filename = secure_filename(f.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(fpath)
        paths[key] = fpath

    features = Features(
        paths['RichTracks'],
        paths['RichTrackPoints'],
        paths['IncomingTrackPoints'],
        host=host)
    if features.error:
        return None, error
    features.predict()

    # before queuing make sure 'rq worker' is running from ./driva !!!
    q = Queue(connection=redis.Redis())
    # close all connections to make features picklable
    features.close_connections()
    # delay saving features call for speed
    q.enqueue(save_features, features)

    # save_features(features)

    return features.pred, features.error


def check_files(files):
    supported_tables = ['RichTracks',
                        'RichTrackPoints', 'IncomingTrackPoints']
    if len(files) == 0:
        return 'no files was received'
    # check if all is ok
    for key in files:
        if key not in supported_tables:
            return key + ': was not expected, expecting only ' + \
                   ', '.join(supported_tables)
        f = files[key]
        if f.filename == '':
            return 'one of the files was not selected'
        elif not allowed_file(f.filename):
            return f.filename + ': extension is not supported'
    return ''


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pred, error = process_files(request.files)
        code = 400 if error else 200
        return render_template('answer.html', error=error, pred=pred), code
    return render_template('index.html')


def get_host(argv):
    global host
    try:
        opts, args = getopt.getopt(argv, "h: ")
    except getopt.GetoptError:
        print('driva.py -h <host_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h",):
            host = arg

    return host


if __name__ == "__main__":
    from werkzeug.contrib.profiler import ProfilerMiddleware

    host = get_host(sys.argv[1:])

    app.config['PROFILE'] = True
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
    app.run(debug=True, host='0.0.0.0', port=80)
    # app.run(host='0.0.0.0', port=1180)
