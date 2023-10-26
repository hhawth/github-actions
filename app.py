from flask import Flask, render_template
from get_fixtures import get_sky_sports_odds, get_fixtures
from stat_getter import get_stats
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)


def get_sky_sports_odds_call():
    print("Getting sky sport's odds")
    get_sky_sports_odds()


def stats_call():
    get_stats()


stats_call()
get_sky_sports_odds_call()

sched = BackgroundScheduler(daemon=True)
sched.add_job(get_sky_sports_odds_call, "interval", minutes=10)
sched.add_job(stats_call, "interval", minutes=360)
sched.start()


@app.route("/")
def hello():
    results = get_fixtures()
    return render_template("home.html", result=results)


if __name__ == "__main__":
    app.run()
