import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import config from './config';
import './App.css';

const createCustomMarkerIcon = () => {
  return new L.Icon({
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    shadowSize: [41, 41]
  });
};

const App = () => {
  const airlineId = '410'; // Example airline ID
  const sourceAirportId = '6156'; // Example source airport ID
  const destinationAirportId = '2952'; // Example destination airport ID

  const [currentDestinationAirportId, setCurrentDestinationAirportId] = useState(destinationAirportId);
  const [route, setRoute] = useState([]);
  const [currentCoords, setCurrentCoords] = useState({ lat: 0, lon: 0 });
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sourceAirportName, setSourceAirportName] = useState('');
  const [destinationAirportName, setDestinationAirportName] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    fetchRoute();
  }, [currentDestinationAirportId]);

  useEffect(() => {
    if (currentCoords.lat && currentCoords.lon) {
      fetchWeather(currentCoords.lat, currentCoords.lon);
    }
  }, [currentCoords]);

  const fetchApprovedChangeRequests = async () => {
    try {
      const response = await axios.get(`${config.backendUrl}/approved-change-requests`, {
        params: {
          airline_id: airlineId
        },
        withCredentials: true
      });
      return response.data.approved_requests;
    } catch (error) {
      console.error("Error fetching approved change requests", error);
      return [];
    }
  };

  const fetchRoute = async () => {
    setLoading(true);
    setError('');
    try {
      console.log('Fetching route...');

      const approvedChangeRequests = await fetchApprovedChangeRequests();

      const approvedRequest = approvedChangeRequests.find(
        (request) => request.source_airline_id === airlineId
      );

      const destinationAirport = approvedRequest ? approvedRequest.nearest_airport_id : currentDestinationAirportId;

      const response = await axios.get(`${config.backendUrl}/route_by_airline`, {
        params: {
          source_airport_id: sourceAirportId,
          destination_airport_id: destinationAirport
        },
        withCredentials: true
      });

      console.log('Route response:', response.data);
      if (response.data.waypoints) {
        setRoute(response.data.waypoints);
        setCurrentCoords(response.data.waypoints[0]);
        setSourceAirportName(response.data.source_name);
        setDestinationAirportName(response.data.destination_name);
      } else {
        console.error("Error fetching route data: No waypoints found");
        setError('No waypoints found for the route');
      }
    } catch (error) {
      if (error.response && error.response.status === 404) {
        setError('Route not found for the specified airports');
      } else {
        console.error("Error fetching route data", error);
        setError('Error fetching route data');
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchWeather = async (lat, lon) => {
    try {
      console.log('Fetching weather...');
      const response = await axios.get(`${config.backendUrl}/weather`, {
        params: { lat, lon }
      });
      console.log('Weather response:', response.data);
      setWeather(response.data);
    } catch (error) {
      console.error("Error fetching weather data", error);
    }
  };

  const requestAlternativeRoute = async () => {
    setLoading(true);
    try {
      console.log('Requesting alternative route...');
      const response = await axios.post(`${config.backendUrl}/requestroute`, {
        current_coords: currentCoords,
        planned_route: route[1],
        Airline_id: airlineId
      }, { withCredentials: true });
      console.log('Alternative route response:', response.data);
      if (response.status === 200) {
        const newRoute = response.data.suggested_airport.approved_route;
        console.log('New route:', newRoute);
        if (newRoute && newRoute.length > 0) {
          setRoute(newRoute);
        } else {
          console.error('Error: No alternative route found');
        }
      } else {
        console.error('Error requesting alternative route:', response.data.error);
      }
    } catch (error) {
      console.error('Error requesting alternative route:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="my-4">Pilot Interface</h1>
      <div className="card mb-4">
        <div className="card-body">
          <h2 className="card-title">Route Information</h2>
          {error && <div className="alert alert-danger">{error}</div>}
          <button className="btn btn-primary mt-4" onClick={fetchRoute} disabled={loading}>
            {loading ? 'Loading...' : 'Fetch Route'}
          </button>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-body">
          <h2 className="card-title">Current Location</h2>
          <MapContainer center={[currentCoords.lat, currentCoords.lon]} zoom={5} style={{ height: '400px' }}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            {route.length > 0 && (
              <>
                <Polyline
                  positions={route.map((coord) => [coord.lat, coord.lon])}
                  color="blue"
                />
                {route.map((coord, index) => (
                  <Marker key={index} position={[coord.lat, coord.lon]} icon={createCustomMarkerIcon()}>
                    <Popup>
                      {index === 0 ? sourceAirportName : index === route.length - 1 ? destinationAirportName : `Waypoint ${index}`}
                    </Popup>
                  </Marker>
                ))}
              </>
            )}
          </MapContainer>
          <button
            className="btn btn-primary mt-4"
            onClick={requestAlternativeRoute}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Request Alternative Route'}
          </button>
        </div>
      </div>

      {weather && (
        <div className="card mb-4">
          <div className="card-body">
            <h2 className="card-title">Current Weather</h2>
            <p><strong>Temperature:</strong> {weather.main.temp} K</p>
            <p><strong>Weather:</strong> {weather.weather[0].description}</p>
            <p><strong>Humidity:</strong> {weather.main.humidity}%</p>
            <p><strong>Wind Speed:</strong> {weather.wind.speed} m/s</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
