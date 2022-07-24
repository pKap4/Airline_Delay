from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

model = load("rand_for.joblib")

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("index.html", href = "static/airplane.png") #jinja used here
    else:
        in_list = [0 for i in range(7)]
        in_list[0] = int(request.form['airline'])
        in_list[1] = int(request.form['flight'])
        in_list[2] = int(request.form['airportFrom'])
        in_list[3] = int(request.form['airportTo'])
        in_list[4] = int(request.form['dayOfWeek'])
        in_list[5] = int(request.form['time'])
        in_list[6] = np.log(int(request.form['length']))

        
        new_pred = model.predict(np.array([in_list]))
        #return render_template("index.html", href = "static\counts_v_Airline.png")
        if new_pred == 0:
            return "<center>The flight will not be delayed.</center>"
        else:
            return "<center>The flight will be delayed.</center>"
        #return in_list[6]

if __name__ == '__main__':
    app.run()
