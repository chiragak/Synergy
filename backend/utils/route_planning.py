import pandas as pd
import os

# Load the nearest airports data into a DataFrame
base_path = os.path.abspath(os.path.dirname(__file__))
nearest_airports_df = pd.read_csv(os.path.join(base_path, '../data/nearest_airports_details.csv'))

def suggest_alternate_route(current_coords, planned_route, mongo):
    # Find the nearest airport
    nearest_airport = mongo.db.nearest_airports.find_one({
        "source_airport_lat": current_coords['lat'],
        "source_airport_lon": current_coords['lon']
    }, sort=[("distance", 1)])

    if nearest_airport:
        return [
            {"lat": current_coords['lat'], "lon": current_coords['lon']},
            {"lat": nearest_airport["nearest_airport_lat"], "lon": nearest_airport["nearest_airport_lon"]}
        ]
    else:
        return planned_route
