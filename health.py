from flask import Flask, render_template, request, jsonify, make_response, json
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

app = Flask(__name__)
xgboost=pickle.load(open('xgboost.pkl','rb'))

@app.route('/', methods=['GET'])
def webapp():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():

    gender = request.form['gender']
    age = request.form['age']
    hypertension = request.form['hypertension']
    heart_disease = request.form['heart_disease']
    ever_married = request.form['ever_married']
    Residence_type = request.form['Residence_type']
    avg_glucose_level = request.form['avg_glucose_level']
    bmi = request.form['bmi']
    work_type = request.form['work_type']
    smoking_status = request.form['smoking_status']
    
    array = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level,
                        bmi, work_type, smoking_status]])
    
    predict = xgboost.predict(array)

    if predict == 0:
        prediction = "No chance of stroke" 
    elif predict == 1:
        prediction = "Chance of stroke"
    
    print(prediction)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)