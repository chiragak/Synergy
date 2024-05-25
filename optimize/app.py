from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
import torch
import torch.nn as nn
import folium
from math import radians, cos, sin, sqrt, atan2
import logging
import base64
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
state_size = 6  # Number of features in the state
action_size = 4  # Example action size: change in direction (0: left, 1: right, 2: up, 3: down)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DQN(state_size, action_size)
model.load_state_dict(torch.load('dqn_model_final.pth'))
model.eval()

# Free API to get weather data
WEATHER_API_KEY = '8b9da067dd72b98bf42ae3cf5e559211'  # Replace with your OpenWeatherMap API key

def get_weather_data(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}'
    response = requests.get(url)
    data = response.json()
    weather_condition = data['weather'][0]['main']
    wind_speed = data['wind']['speed']
    visibility = data.get('visibility', 10000) / 1000  # Convert visibility to km
    weather_condition_encoded = {'Clear': 0, 'Rain': 1, 'Storm': 2}.get(weather_condition, 0)
    return weather_condition_encoded, wind_speed, visibility

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        current_lat = float(data['current_lat'])
        current_lon = float(data['current_lon'])
        altitude = float(data['altitude'])
        
        weather_condition, wind_speed, visibility = get_weather_data(current_lat, current_lon)
        
        state = np.array([current_lon, current_lat, altitude, weather_condition, wind_speed, visibility], dtype=np.float32)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_values = model(state)
        
        action = np.argmax(action_values.numpy())
        
        return jsonify({'action': int(action)})
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return jsonify({'error': 'Invalid input data'}), 400
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/plot_path', methods=['POST'])
def plot_path():
    data = request.json
    try:
        source_lat = float(data['source_lat'])
        source_lon = float(data['source_lon'])
        dest_lat = float(data['dest_lat'])
        dest_lon = float(data['dest_lon'])

        distance = haversine(source_lat, source_lon, dest_lat, dest_lon)
        avg_speed_kmh = 900  # Average speed in km/h
        avg_fuel_consumption_lph = 2500  # Average fuel consumption in liters per hour

        flight_time = distance / avg_speed_kmh  # in hours
        fuel_consumption = flight_time * avg_fuel_consumption_lph  # in liters

        # Create map
        map_ = folium.Map(location=[(source_lat + dest_lat) / 2, (source_lon + dest_lon) / 2], zoom_start=4)
        folium.Marker([source_lat, source_lon], popup='Source', icon=folium.Icon(color='green')).add_to(map_)
        folium.Marker([dest_lat, dest_lon], popup='Destination', icon=folium.Icon(color='red')).add_to(map_)
        folium.PolyLine([(source_lat, source_lon), (dest_lat, dest_lon)], color='blue').add_to(map_)

        # Save map to HTML
        map_html = 'map.html'
        map_.save(map_html)

        # Encode HTML map to base64
        with open(map_html, 'r') as f:
            map_base64 = base64.b64encode(f.read().encode()).decode()

        return jsonify({
            'distance': distance,
            'flight_time': flight_time,
            'fuel_consumption': fuel_consumption,
            'map_base64': map_base64
        })
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return jsonify({'error': 'Invalid input data'}), 400
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
