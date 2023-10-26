from flask import Flask, render_template
from get_fixtures import get_sky_sports_odds, get_fixtures
from stat_getter import get_stats
from apscheduler.schedulers.background import BackgroundScheduler
import time

LAST_STATS_CALL = None

app = Flask(__name__)


def get_sky_sports_odds_call():
    get_sky_sports_odds()


def stats_call():
    global LAST_STATS_CALL
    get_stats()
    LAST_STATS_CALL = time.time()


stats_call()
get_sky_sports_odds_call()


@app.route("/")
def hello():
    if time.time() - LAST_STATS_CALL > 6000:
        stats_call()
    results = get_fixtures()
    return render_template("home.html", result=results)


if __name__ == "__main__":
    app.run()
