from flask import Flask, render_template
from get_fixtures import get_fixtures
from stat_getter import get_stats
from apscheduler.schedulers.background import BackgroundScheduler

FIXTURES = None
app = Flask(__name__)


def fixtures_call():
    global FIXTURES
    FIXTURES = get_fixtures()


def stats_call():
    get_stats()


stats_call()
fixtures_call()

sched = BackgroundScheduler(daemon=True)
sched.add_job(fixtures_call, "interval", minutes=360)
sched.add_job(fixtures_call, "interval", minutes=360)
sched.start()


@app.route("/")
def hello():
    results = FIXTURES
    return render_template("home.html", result=results)


if __name__ == "__main__":
    app.run()
