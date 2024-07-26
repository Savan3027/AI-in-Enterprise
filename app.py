import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

label = ['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish']

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input features from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Predicted Species: {label[output]}')

if __name__ == "__main__":
    app.run(debug=True)
