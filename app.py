from flask import Flask, redirect, url_for, render_template, request;
import joblib;
import numpy as np;
import pandas as pd;

app = Flask(__name__)

@app.route("/", methods = ["POST","GET"])
def home():
    if request.method == "POST":
        age = request.form["age"]
        weight = request.form["weight"]

        model = joblib.load("regr.pkl")
        x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
        result = model.predict(x)[0]

        return render_template("main.html", result = result)
    else:
        return render_template("main.html")


# @app.route("/calculate")
# def calculate():
#     return render_template("calculate.html")

if __name__ == "__main__":
    app.debug = True
    app.run()