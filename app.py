from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json
    
    # Extract the details from the data
    car_details = [
        float(data['Year']),
        float(data['Selling_Price']),
        float(data['Present_Price']),
        float(data['Kms_Driven']),
        float(data['Fuel_Type']),
        float(data['Seller_Type']),
        float(data['Transmission']),
        float(data['Owner'])
    ]
    
    input_data = np.array(car_details).reshape(1, -1)
    prediction = model.predict(input_data)
    response = {'prediction': round(prediction[0], 2)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
