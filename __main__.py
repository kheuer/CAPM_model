from datetime import timedelta

from flask import Flask, redirect, url_for, render_template, request, flash, session

from auxiliary_functions import *

app = Flask("__main__")
app.secret_key = "ghjvfcd87tfcdtrergef5t45er"
app.permanent_session_lifetime = timedelta(minutes=30)
#app.config["SESSION_TYPE"] = "filesystem"

portfolios_manager = PortfolioManager()


@app.route("/")
def home():
    init_session()
    ticker_symbols = list(session["assets"].keys())

    min_position_weight = session["portfolio_settings"]["min_position_weight"]
    max_position_weight = session["portfolio_settings"]["max_position_weight"]

    if not ticker_symbols:
        return render_template("index.html",
                               username=session["username"],
                               assets=session["assets"],
                               portfolio_sd_er=[],
                               scatter_colors=[],
                               portfolios_description=[],
                               portfolio_weights=[],
                               risk_free_rate=session["risk_free_weight"],
                               risk_free_weights=[],
                               portfolio_index_lookup_array=[]
                               )

    try:
        portfolios_data = portfolios_manager.get_data(ticker_symbols, min_position_weight, max_position_weight)
    except AssertionError:
        flash("Error while calculating portfolios, this is likely caused by impossible criteria for min/max weights")
        return render_template("index.html",
                               username=session["username"],
                               assets=session["assets"],
                               portfolio_sd_er=[],
                               scatter_colors=[],
                               portfolios_description=[],
                               portfolio_weights=[],
                               risk_free_weights=[],
                               risk_free_rate=session["risk_free_weight"],
                               portfolio_index_lookup_array=[]
                               )
    except ApiRateError:
        flash("Internal Error while trying to call current price data, please retry.")
        return render_template("index.html",
                               username=session["username"],
                               assets=session["assets"],
                               portfolio_sd_er=[],
                               scatter_colors=[],
                               portfolios_description=[],
                               portfolio_weights=[],
                               risk_free_weights=[],
                               risk_free_rate=session["risk_free_rate"],
                               portfolio_index_lookup_array=[]
                               )

    session.modified = True
    return render_template("index.html",
                           username=session["username"],
                           assets=session["assets"],
                           portfolio_sd_er=portfolios_data["data"],
                           scatter_colors=portfolios_data["colors"],
                           portfolios_description=portfolios_data["description"],
                           portfolio_weights=portfolios_data["weights"],
                           risk_free_weights=portfolios_data["percentages_risk_free"],
                           risk_free_rate=session["risk_free_rate"],
                           portfolio_index_lookup_array=portfolios_data["lookup_array"],
                           )


@app.route("/login/", methods=["POST", "GET"])
def login():
    init_session()
    if request.method == "POST":
        session.modified = True
        username = request.form["username"]
        password = request.form["password"]

        if check_credentials(username, password):
            session["username"] = username
            session["authenticated"] = True
            flash('You were successfully logged in')
            return redirect(url_for("home"))
        else:
            flash("Incorrect Login Information")
            return render_template("login.html")
    else:
        if session["authenticated"]:
            flash("Already logged In.")
            return redirect(url_for("home"))
        else:
            return render_template("login.html")


@app.route("/logout/")
def logout():
    flash("You were logged out.")
    session["username"] = "Guest"
    session["authenticated"] = False
    return redirect(url_for("home"))


@app.route("/user/", methods=["GET", "POST"])
def user():
    init_session()
    if request.method == "POST":
        if request.form["type"] == "change_settings":
            min_position_weight = request.form["min_position_weight"]
            max_position_weight = request.form["max_position_weight"]
            cash_value_portfolio = request.form["cash_value_portfolio"]
            risk_free_rate = request.form["risk_free_rate"]

            n_positions = len(session["assets"])
            try:
                min_position_weight = float(min_position_weight)
                max_position_weight = float(max_position_weight)
                cash_value_portfolio = float(cash_value_portfolio)
                risk_free_rate = float(risk_free_rate)
            except ValueError:
                flash(f"Error: all entries must be numerical")
                return render_template("user.html")

            error_message = None
            if min_position_weight > 1 or max_position_weight > 1:
                error_message = f"Error: portfolio weights cannot exceed 100% (1)"
            elif min_position_weight < 0 or max_position_weight < 0:
                error_message = f"Error: portfolio weights cannot be less than 0% (0)"
            elif min_position_weight > max_position_weight:
                error_message = f"Error: minimum position weight cannot exceed maximum position weight"
            elif cash_value_portfolio < 1000:
                error_message = "Error: Portfolio value cannot be less than 1000 USD"
            elif risk_free_rate > 0.4:
                error_message = "Error: Risk free rate cannot exceed 40% (0.4)"
            elif risk_free_rate < -0.2:
                error_message = "Error: Risk free rate cannot be less than -20% (-0.2)"
            elif n_positions * max_position_weight < 0.999999999:
                error_message = f"Error: {n_positions} Selected Assets and Maximum Position Weight of {max_position_weight} are not compatible"

            if error_message:
                flash(error_message)
                return render_template("user.html",
                                       saved_portfolios=settings_container[session["username"]]["portfolios"]
                                       )

            session["portfolio_settings"] = {
                "min_position_weight": min_position_weight,
                "max_position_weight": max_position_weight,
                "portfolio_value": cash_value_portfolio
            }
            settings_container[session["username"]]["min_position_weight"] = min_position_weight
            settings_container[session["username"]]["max_position_weight"] = max_position_weight
            settings_container[session["username"]]["portfolio_value"] = cash_value_portfolio
            settings_container[session["username"]]["risk_free_rate"] = risk_free_rate

            with open("settings.yaml", "w") as file:
                yaml.dump(settings_container, file)

            flash("saved settings")
        elif request.form["type"] == "load_portfolio":
            portfolios = settings_container[session["username"]]["portfolios"]
            assert portfolios
            for portfolio in portfolios:
                if portfolio["name"] == request.form["portfolio_name"]:
                    break   # break when the current portfolio is the one with the queried name
            error_message = None
            if not settings_container[session["username"]]["portfolios"]:
                error_message = "No portfolios saved."
            elif not portfolio["name"] == request.form["portfolio_name"]:
                error_message = "The queried portfolio is not saved"
            if error_message:
                flash(error_message)
                return render_template("user.html",
                                       saved_portfolios=settings_container[session["username"]]["portfolios"]
                                       )
            else:
                session["portfolio_settings"]["min_position_weight"] = portfolio["min_position_weight"]
                session["portfolio_settings"]["max_position_weight"] = portfolio["max_position_weight"]
                session["assets"] = {ticker_symbol: get_asset_descriptor(ticker_symbol) for ticker_symbol in portfolio["positions"]}

                flash("loaded portfolio")
        session.modified = True
        return redirect(url_for("home"))
    else:
        return render_template("user.html",
                               saved_portfolios=settings_container[session["username"]]["portfolios"]
                               )


@app.route("/search/", methods=["GET", "POST"])
def search():
    init_session()
    if request.method == "POST":
        query = request.form["q"]
        print("query", query)
        if query in all_ticker_symbols:
            if query in session["assets"]:
                session["assets"].pop(query)
            else:
                session["assets"][query] = get_asset_descriptor(query)
        else:
            flash(f"could not find ticker symbol: {query}")
    session.modified = True
    return redirect(url_for("home"))


@app.route("/portfolio/<sd>/<er>")
def show_portfolio(sd, er):
    init_session()
    ticker_symbols = list(session["assets"].keys())
    min_position_weight = session["portfolio_settings"]["min_position_weight"]
    max_position_weight = session["portfolio_settings"]["max_position_weight"]
    portfolios_data = portfolios_manager.get_data(ticker_symbols, min_position_weight, max_position_weight)

    try:
        sd = float(sd)
        er = float(er)
    except ValueError:
        flash("Internal Error: sd and er pair is not numerical")

    lookup_val = str(sd)+str(er)
    try:
        portfolio_index = portfolios_data["lookup_array"].index(lookup_val)
    except (ValueError, KeyError):
        flash("Internal Error: sd and er pair is not saved")
        return render_template(url_for("home"))

    weights = list(portfolios_data["weights"][(sd, er)])
    cash = session["portfolio_settings"]["portfolio_value"]

    n_shares_dict = portfolios_data["instance"].weights_to_positions(weights, cash)
    n_shares = []
    for ticker_symbol in portfolios_data["ticker_symbols"]:
        n_shares.append(n_shares_dict[ticker_symbol])

    return render_template("portfolio.html",
                           portfolio_index=portfolio_index,
                           weights_data=weights,
                           risk_free_weight=portfolios_data["percentages_risk_free"][(sd, er)],
                           standard_deviation=sd,
                           expected_return=er,
                           risk_free_rate=session["risk_free_rate"],
                           portfolio_value=cash,
                           portfolio_description=portfolios_data["description"][portfolio_index],
                           company_names=portfolios_data["company_names"],
                           ticker_symbols=portfolios_data["ticker_symbols"],
                           n_shares=[int(x) for x in n_shares],
                           n_positions=len(portfolios_data["ticker_symbols"])
                           )

@app.errorhandler(404) 
def invalid_route(e): 
    return render_template("404.html")

@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html")


@app.route("/clear_session/")
def clear_session():
    session.clear()
    return redirect(url_for("home"))

@app.route("/session/")
def show_session():
    return session

def init_session():
    session.permanent = False
    session["SameSite"] = None

    if "authenticated" not in session:
        session["authenticated"] = False
    if "assets" not in session:
        session["assets"] = {"AAPL": get_asset_descriptor("AAPL"),
                             "AMZN": get_asset_descriptor("AMZN"),
                             "MSFT": get_asset_descriptor("MSFT")
                             }

    if not session["authenticated"]:
        session["username"] = "Guest"

    user_settings = settings_container[session["username"]]
    print("user settings", user_settings)

    if "portfolio_settings" not in session:
        session["portfolio_settings"] = {
            "min_position_weight": user_settings["min_position_weight"],
            "max_position_weight": user_settings["max_position_weight"],
            "portfolio_value": user_settings["portfolio_value"]
        }

    if "risk_free_rate" not in session:
        session["risk_free_rate"] = user_settings["risk_free_rate"]

    if "saved_portfolios" not in session:
        print(user_settings)
        print("saved", user_settings["portfolios"])
        session["saved_portfolios"] = user_settings["portfolios"]

    session.modified = True



if __name__ == "__main__":
    print("Loading Data.")
    portfolios_manager.get_data(["AMZN", "AAPL", "MSFT"], 0.1, 0.9)
    print("Starting Server.")
    app.run(host="0.0.0.0", port=5000, debug=False)
