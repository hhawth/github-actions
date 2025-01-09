import logging
from flask import Flask, render_template

from get_fixtures import get_fixtures

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

LAST_STATS_CALL = None

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/")
def main():
    results = get_fixtures()
    return render_template("home.html", matches=results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
