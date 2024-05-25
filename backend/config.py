from urllib.parse import quote_plus

# Replace these with your actual MongoDB username and password
username = quote_plus('chiragajekar')
password = quote_plus('Chirag@123')

# Construct the MongoDB URI
MONGO_URI = f"mongodb+srv://{username}:{password}@flightnavcluster.qvoys0n.mongodb.net/flight_navigation?retryWrites=true&w=majority&appName=FlightNavCluster"

OPENWEATHER_API_KEY = "b2d300eac13dfa8a3b4504d6532c74f0"

# Secret key for session management and other security-related needs
SECRET_KEY = "a_random_secret_key"  # Replace with a secure random key in production
