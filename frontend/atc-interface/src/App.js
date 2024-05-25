import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import config from './config';

const ATCInterface = () => {
  const [changeRequests, setChangeRequests] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  useEffect(() => {
    fetchChangeRequests();
  }, []);

  const fetchChangeRequests = async () => {
    try {
      const response = await axios.get(`${config.backendUrl}/change-requests`);
      setChangeRequests(response.data.change_requests || []);
    } catch (error) {
      console.error('Error fetching change requests:', error);
    }
  };

  const handleApproveRejectRequest = async (requestId, status) => {
    try {
      const request = changeRequests.find(req => req._id === requestId);
      const response = await axios.post(`${config.backendUrl}/approve-change-route`, {
        request_id: requestId,
        status: status,
        selected_airport: {
          airport_id: request.nearest_airport_id,
          airport_name: request.nearest_airport_name,
          airport_city: request.nearest_airport_city,
          airport_country: request.nearest_airport_country,
          airport_lat: request.nearest_airport_latitude,
          airport_lon: request.nearest_airport_longitude,
        }
      });
      console.log(response.data.message);
      fetchChangeRequests();
    } catch (error) {
      console.error('Error approving/rejecting change request:', error);
    }
  };

  const handleSearch = async () => {
    try {
      const response = await axios.get(`${config.backendUrl}/search_airports`, {
        params: { query: searchQuery }
      });
      setSearchResults(response.data.airports || []);
    } catch (error) {
      console.error('Error searching airports:', error);
    }
  };

  return (
    <div className="container">
      <h1 className="my-4">ATC Interface</h1>

      <div className="card mb-4">
        <div className="card-body">
          <h2 className="card-title">Search Airports</h2>
          <div className="input-group mb-3">
            <input
              type="text"
              className="form-control"
              placeholder="Search by airport name, city, or country"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="input-group-append">
              <button className="btn btn-primary" onClick={handleSearch}>Search</button>
            </div>
          </div>
          {searchResults.length > 0 && (
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>City</th>
                  <th>Country</th>
                  <th>Latitude</th>
                  <th>Longitude</th>
                </tr>
              </thead>
              <tbody>
                {searchResults.map((airport) => (
                  <tr key={airport._id}>
                    <td>{airport.Name}</td>
                    <td>{airport.City}</td>
                    <td>{airport.Country}</td>
                    <td>{airport.Latitude}</td>
                    <td>{airport.Longitude}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-body">
          <h2 className="card-title">Change Requests</h2>
          {changeRequests.length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Airline ID</th>
                  <th>Nearest Airport</th>
                  <th>City</th>
                  <th>Country</th>
                  <th>Latitude</th>
                  <th>Longitude</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {changeRequests.map((request) => (
                  <tr key={request._id}>
                    <td>{request.source_airline_id}</td>
                    <td>{request.nearest_airport_name}</td>
                    <td>{request.nearest_airport_city}</td>
                    <td>{request.nearest_airport_country}</td>
                    <td>{request.nearest_airport_latitude}</td>
                    <td>{request.nearest_airport_longitude}</td>
                    <td>
                      <button
                        className="btn btn-success mr-2"
                        onClick={() => handleApproveRejectRequest(request._id, 'approved')}
                      >
                        Approve
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={() => handleApproveRejectRequest(request._id, 'rejected')}
                      >
                        Reject
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p>No change requests found.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ATCInterface;
