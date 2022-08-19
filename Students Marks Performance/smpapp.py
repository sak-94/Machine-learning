import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

smpapp = Flask(__name__)

@smpapp.route('/')
def Home():
    return render_template('students marks perfomance.html')

@smpapp.route('/predict', methods=['POST','GET'])
def results():
    number_courses = float(request.form['number_courses'])
    time_study = float(request.form['time_study'])
##    Marks = float(request.form['Marks'])


    X = np.array([[number_courses,time_study]])
    model1 = pickle.load(open('model1.pkl', 'rb'))
    Y_predict = model1.predict(X)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    smpapp.run(debug = True, port = 1011)