from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load the pre-trained model
model_file = 'models/deploy_DT.pkl'
with open(model_file, 'rb') as file:
    model1 = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_data=[]
    
    input_data.append(float(request.form['MDVP_Fo']))
    input_data.append(float(request.form['MDVP_Fhi']))
    input_data.append(float(request.form['MDVP_Flo']))
    input_data.append(float(request.form['MDVP_Jitter']))
    input_data.append(float(request.form['MDVP_Jitter_Abs']))
    input_data.append(float(request.form['MDVP_RAP']))
    input_data.append(float(request.form['MDVP_PPQ']))
    input_data.append(float(request.form['Jitter_DDP']))
    input_data.append(float(request.form['MDVP_Shimmer']))
    input_data.append(float(request.form['MDVP_Shimmer_dB']))
    input_data.append(float(request.form['Shimmer_APQ3']))
    input_data.append(float(request.form['Shimmer_APQ5']))
    input_data.append(float(request.form['MDVP_APQ']))
    input_data.append(float(request.form['Shimmer_DDA']))
    input_data.append(float(request.form['NHR']))
    input_data.append(float(request.form['HNR']))
    
    input_data.append(float(request.form['RPDE']))
    input_data.append(float(request.form['DFA']))
    input_data.append(float(request.form['spread1']))
    input_data.append(float(request.form['spread2']))
    input_data.append(float(request.form['D2']))
    input_data.append(float(request.form['PPE']))
    
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array for model 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model1.predict(input_data_reshaped)

    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)