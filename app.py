import logging

from flask import Flask, render_template
from flask_apscheduler import APScheduler

from get_fixtures import get_fixtures, get_fixtures_and_odds
from stat_getter import get_stats

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

LAST_STATS_CALL = None

app = Flask(__name__)


class Config:
    SCHEDULER_API_ENABLED = True


app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


@scheduler.task("interval", id="do_job_2", hours=6, misfire_grace_time=900)
def get_fixtures_and_odds_call():
    get_fixtures_and_odds()


@scheduler.task("interval", id="do_job_1", hours=6, misfire_grace_time=900)
def stats_call():
    LOGGER.info("calling stats")
    get_stats()


stats_call()
get_fixtures_and_odds_call()


@app.route("/")
def hello():
    results = get_fixtures()
    return render_template("home.html", matches=results)


if __name__ == "__main__":
    app.run()
