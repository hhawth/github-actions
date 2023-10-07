from flask import Flask, render_template
from get_fixtures import get_fixtures
app = Flask(__name__)

@app.route("/")
def hello():
    results = get_fixtures()
    return render_template("home.html", result = results)