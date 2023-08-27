from flask import Flask
from datetime import datetime
import sys
import application_context
import multiprocessing
import psutil
import os

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/time")
def what_is_time():
    return str(datetime.now())


@app.route("/counter")
def next_val():
    application_context.global_counter = application_context.global_counter + 1
    return str(application_context.global_counter)


@app.route("/about")
def me_api():

    return {
        "platform": sys.platform,
        "version": sys.version,
        "cores": psutil.cpu_count(),
        "coresUsage": psutil.cpu_percent(),
        "diskUsage": psutil.disk_usage("/"),
    }


if __name__ == '__main__':
    app.run(debug=True, port=sys.argv[1])
