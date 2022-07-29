from quart import Blueprint, redirect, render_template

# ---------------

impBP = Blueprint('imp', __name__, template_folder='templates', static_folder='static')

# ---------------

@impBP.route("/")
async def hello():
    return await render_template('home.html')