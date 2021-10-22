import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, redirect, render_template, json, jsonify
import requests

sc = StandardScaler()
app = Flask(__name__)
model = pickle.load(open('randomforestmodel.pickle', 'rb'))
dataset = pd.read_csv('cleaned_data.csv', index_col=0)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method=="POST":
        time_in_sec =  float(request.form['time_in_sec'])
        Acceleration_vertical_axis = float(request.form['Acceleration_vertical_axis'])
        Acceleration_lateral_axis  = float(request.form['Acceleration_lateral_axis'])
        id_of_antenna_reading = int(request.form['id_of_antenna_reading'])
        signal_strength_indicator = float(request.form['signal_strength_indicator'])
        Phase = float(request.form['Phase'])
        Frequency = float(request.form['Frequency'])
        
        #load the pickle file
        filename = "randomforestmodel.pickle"
        loaded_model = pickle.load(open(filename, 'rb'))
        
        data = np.array([[time_in_sec, Acceleration_vertical_axis, 
                          Acceleration_lateral_axis,id_of_antenna_reading,
                          signal_strength_indicator, Phase, Frequency]])
        print(data)
        
        data = sc.fit_transform(data)
        my_prediction = loaded_model.predict(data)
        #get the result template
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)                    

