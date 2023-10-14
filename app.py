import joblib
import numpy as np
from flask import Flask, request, jsonify
import sklearn
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load the trained model
model = joblib.load('StockPrediction_model.pkl')


@app.route('/stockPredict', methods=['POST'])
def predict():
    try:
        # Get the list of lists of stocks from the request
        data = request.json

        # Ensure that the input is a list of lists
        if not isinstance(data, list):
            return jsonify({"error": "Input should be a list of lists of stocks"}), 400

        # Convert the input data into a numpy array
        input_data = np.array(data)

        # Make predictions for all sets of features
        predictions = model.predict(input_data)

        # Cast the predictions to integers
        predictions = predictions.astype(int)

        # Return the predictions as a JSON response
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/status', methods=['GET'])
def server_status():
    return 'Server is running'


if __name__ == '__main__':
    app.run(debug=True)
