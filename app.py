from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define the same LSTM model structure used in training
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Load the trained model
model = LSTMForecast()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Forecasting function
def predict_future(input_series, forecast_horizon):
    data = input_series.copy()
    predictions = []
    for _ in range(forecast_horizon):
        seq = torch.FloatTensor(data[-12:])  # Using last 12 time steps
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            prediction = model(seq)
            pred_value = prediction.item()
            data.append(pred_value)
            predictions.append(pred_value)
    return predictions

# Endpoint
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        input_json = request.get_json()
        series = input_json['series']
        horizon = input_json['forecast_horizon']
        forecast = predict_future(series, horizon)
        return jsonify({"forecast": forecast})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

