from quart import Quart, redirect
import os

# New
import quart.flask_patch

config = {
    "DEBUG": True,                # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

# ---------------

app = Quart(__name__, static_folder=None)

# New
app.config.from_mapping(config)

# ---------------

from impaintingWeb import impBP
app.register_blueprint(impBP,url_prefix='/imp')

# ---------------

# Default route
@app.route("/")
def default():
    return redirect("/imp")

# ---------------

if __name__ == "__main__":
	app.run(host="localhost", port=5000, debug=True)