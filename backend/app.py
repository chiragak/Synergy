import pandas as pd
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from bson import ObjectId
import os
from utils.weather import get_weather
from utils.route_planning import suggest_alternate_route
from flask_cors import CORS, cross_origin
import config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": ["http://localhost:3001", "http://localhost:3000"]}}, supports_credentials=True)
app.config["MONGO_URI"] = config.MONGO_URI
app.config['SECRET_KEY'] = config.SECRET_KEY

@app.route('/')
@cross_origin()
def index():
    return "Hello, World!"

mongo = PyMongo(app)

@app.route('/api/initdb', methods=['POST'])
def init_db():
    try:
        # Drop existing collections
        mongo.db.airlines.drop()
        mongo.db.airplanes.drop()
        mongo.db.airports.drop()
        mongo.db.routes.drop()
        mongo.db.change_requests.drop()
        mongo.db.nearest_airports.drop()

        # Load data from CSV files
        base_path = os.path.abspath(os.path.dirname(__file__))
        airlines = pd.read_csv(os.path.join(base_path, 'data/airlines.csv'))
        airplanes = pd.read_csv(os.path.join(base_path, 'data/airplanes.csv'))
        airports = pd.read_csv(os.path.join(base_path, 'data/airports.csv'))
        routes = pd.read_csv(os.path.join(base_path, 'data/routes.csv'))
        nearest_airports = pd.read_csv(os.path.join(base_path, 'data/nearest_airports_details.csv'))

        # Convert data to dictionaries
        airlines_dict = airlines.to_dict(orient='records')
        airplanes_dict = airplanes.to_dict(orient='records')
        airports_dict = airports.to_dict(orient='records')
        routes_dict = routes.to_dict(orient='records')
        nearest_airports_dict = nearest_airports.to_dict(orient='records')

        # Insert data into MongoDB
        mongo.db.airlines.insert_many(airlines_dict)
        mongo.db.airplanes.insert_many(airplanes_dict)
        mongo.db.airports.insert_many(airports_dict)
        mongo.db.routes.insert_many(routes_dict)
        mongo.db.nearest_airports.insert_many(nearest_airports_dict)

        return jsonify({"message": "Database initialized successfully!"})

    except Exception as e:
        print(f"Error initializing database: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/weather', methods=['GET'])
def weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    weather_data = get_weather(lat, lon, config.OPENWEATHER_API_KEY)
    return jsonify(weather_data)

def get_airport_coords(airport_id):
    try:
        airport = mongo.db.airports.find_one({"Airport ID": int(airport_id)})
        if airport:
            return {"lat": airport["Latitude"], "lon": airport["Longitude"]}
        else:
            print(f"Airport not found for ID: {airport_id}")
            return None
    except Exception as e:
        print(f"Error getting airport coordinates for ID {airport_id}: {e}")
        return None

@app.route('/api/route_by_airline', methods=['GET'])
def get_route_by_airline():
    try:
        source_airport_id = request.args.get('source_airport_id')
        destination_airport_id = request.args.get('destination_airport_id')

        logger.info(f"Querying airports with Source Airport ID: {source_airport_id}, Destination Airport ID: {destination_airport_id}")

        source_airport = mongo.db.airports.find_one({"Airport ID": int(source_airport_id)})
        destination_airport = mongo.db.airports.find_one({"Airport ID": int(destination_airport_id)})

        if source_airport and destination_airport:
            source_coords = {"lat": source_airport["Latitude"], "lon": source_airport["Longitude"]}
            destination_coords = {"lat": destination_airport["Latitude"], "lon": destination_airport["Longitude"]}
            waypoints = [source_coords, destination_coords]
            logger.info(f"Route found: Source - {source_airport['Name']}, Destination - {destination_airport['Name']}")
            return jsonify({
                "waypoints": waypoints,
                "source_name": source_airport["Name"],
                "destination_name": destination_airport["Name"]
            })
        else:
            logger.error("One or both airports not found")
            return jsonify({"error": "One or both airports not found"}), 404
    except Exception as e:
        logger.error(f"Error fetching route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




@app.route('/api/requestroute', methods=['POST'])
def requestroute():
    try:
        data = request.json
        source_airline_id = data['Airline_id']
        planned_route = data['planned_route']

        # Find the current airport based on the planned route coordinates
        airport_response = mongo.db.airports.find_one({
            "Latitude": planned_route['lat'],
            "Longitude": planned_route['lon']
        })

        if not airport_response:
            return jsonify({"error": "No airport found at the provided coordinates"}), 404

        airport_name = airport_response['Name']

        # Find the nearest airport to the current airport sorted by distance
        nearest_airport = mongo.db.nearest_airports.find_one({
            "Airport": airport_name
        }, sort=[("Distance (km)", 1)])

        if not nearest_airport:
            return jsonify({"error": "No nearest airport found for the provided airport"}), 404

        nearest_airport_data = mongo.db.airports.find_one({
            "Name": nearest_airport['Nearest Airport']
        })

        if nearest_airport_data:
            change_request = {
                "source_airline_id": source_airline_id,
                "nearest_airport_id": str(nearest_airport_data["Airport ID"]),
                "nearest_airport_name": nearest_airport_data["Name"],
                "nearest_airport_city": nearest_airport_data["City"],
                "nearest_airport_country": nearest_airport_data["Country"],
                "nearest_airport_latitude": nearest_airport_data["Latitude"],
                "nearest_airport_longitude": nearest_airport_data["Longitude"]
            }
            change_request_id = mongo.db.change_requests.insert_one(change_request).inserted_id
            change_request['_id'] = str(change_request_id)  # Convert ObjectId to string
            return jsonify({"message": "Change request submitted successfully", "suggested_airport": change_request})
        else:
            return jsonify({"error": "No nearest airport details found"}), 404

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/change-requests', methods=['GET'])
def change_requests():
    change_requests = list(mongo.db.change_requests.find())
    for request in change_requests:
        request['_id'] = str(request['_id'])  # Convert ObjectId to string for JSON serialization
    return jsonify({"change_requests": change_requests})

@app.route('/api/approve-change-route', methods=['POST'])
def approve_change_route():
    try:
        data = request.json
        request_id = data.get('request_id')
        status = data.get('status')

        if not request_id or not status:
            return jsonify({"error": "Request ID and status are required"}), 400

        change_request = mongo.db.change_requests.find_one({"_id": ObjectId(request_id)})
        if change_request:
            # Update the change request with the status
            mongo.db.change_requests.update_one(
                {"_id": ObjectId(request_id)},
                {"$set": {"status": status}}
            )
            return jsonify({"message": f"Change request {status} successfully"})
        else:
            return jsonify({"error": "Change request not found"}), 404
    except Exception as e:
        print(f"Error processing change request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/all_routes', methods=['GET'])
def all_routes():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        routes = mongo.db.routes.find().skip((page - 1) * per_page).limit(per_page)
        response = []
        for route in routes:
            source_coords = get_airport_coords(route["Source airport ID"])
            destination_coords = get_airport_coords(route["Destination airport ID"])
            if source_coords and destination_coords:
                response.append({
                    "source_name": route["Source airport"],
                    "destination_name": route["Destination airport"],
                    "source_coords": source_coords,
                    "destination_coords": destination_coords
                })
            else:
                print(f"Coordinates not found for route: {route}")
        return jsonify({"routes": response})
    except Exception as e:
        print(f"Error fetching all routes: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/nearest-airports', methods=['GET'])
def nearest_airports():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        nearest_airports = list(mongo.db.nearest_airports.find({
            "source_airport_lat": lat,
            "source_airport_lon": lon
        }).sort("distance", 1).limit(5))
        return jsonify({"nearest_airports": nearest_airports})
    except Exception as e:
        print(f"Error fetching nearest airports: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/approved-change-requests', methods=['GET'])
def approved_change_requests():
    try:
        airline_id = request.args.get('airline_id')
        if not airline_id:
            raise ValueError("Airline ID is required")

        approved_requests = list(mongo.db.change_requests.find({
            "source_airline_id": airline_id,
            "status": "approved"
        }))

        if not approved_requests:
            logger.info(f"No approved change requests found for airline_id: {airline_id}")
            return jsonify({"approved_requests": []})

        for req in approved_requests:
            req['_id'] = str(req['_id'])  # Convert ObjectId to string for JSON serialization

        return jsonify({"approved_requests": approved_requests})

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error fetching approved change requests: {e}")
        return jsonify({"error": "Internal Server Error"}), 500



@app.route('/api/search_airports', methods=['GET'])
def search_airports():
    try:
        query = request.args.get('query', '').strip()

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Exact match search (case-insensitive)
        exact_matches = mongo.db.airports.find({
            "$or": [
                {"Name": {"$regex": f"^{query}$", "$options": "i"}},
                {"City": {"$regex": f"^{query}$", "$options": "i"}},
                {"Country": {"$regex": f"^{query}$", "$options": "i"}}
            ]
        })

        # Partial match search
        partial_matches = mongo.db.airports.find({
            "$or": [
                {"Name": {"$regex": query, "$options": "i"}},
                {"City": {"$regex": query, "$options": "i"}},
                {"Country": {"$regex": query, "$options": "i"}}
            ]
        })

        results = []

        # Add exact matches first
        for airport in exact_matches:
            airport['_id'] = str(airport['_id'])  # Convert ObjectId to string for JSON serialization
            results.append(airport)

        # Add partial matches, avoiding duplicates based on airport name
        for airport in partial_matches:
            airport_name = airport['Name']
            if not any(result['Name'] == airport_name for result in results):
                airport['_id'] = str(airport['_id'])  # Convert ObjectId to string for JSON serialization
                results.append(airport)

        return jsonify({"airports": results})

    except Exception as e:
        print(f"Error searching airports: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
                