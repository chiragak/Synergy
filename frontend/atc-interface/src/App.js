import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import config from './config';

const ATCInterface = () => {
  const [changeRequests, setChangeRequests] = useState([]);

  useEffect(() => {
    fetchChangeRequests();
  }, []);

  const fetchChangeRequests = async () => {
    try {
      const response = await axios.get(`${config.backendUrl}/change-requests`);
      setChangeRequests(response.data.change_requests);
    } catch (error) {
      console.error('Error fetching change requests:', error);
    }
  };

  const handleApproveRejectRequest = async (requestId, status) => {
    try {
      const response = await axios.post(`${config.backendUrl}/approve-change-route`, {
        request_id: requestId,
        selected_airport: {
          airport_id: changeRequests.find(req => req._id === requestId).nearest_airport_id,
          airport_name: changeRequests.find(req => req._id === requestId).nearest_airport_name,
          airport_city: changeRequests.find(req => req._id === requestId).nearest_airport_city,
          airport_country: changeRequests.find(req => req._id === requestId).nearest_airport_country,
          airport_lat: changeRequests.find(req => req._id === requestId).nearest_airport_latitude,
          airport_lon: changeRequests.find(req => req._id === requestId).nearest_airport_longitude,
        },
        status: status,
      });
      console.log(response.data.message);
      fetchChangeRequests();
    } catch (error) {
      console.error('Error approving/rejecting change request:', error);
    }
  };

  return (
    <div className="container">
      <h1 className="my-4">ATC Interface</h1>
      <div className="card mb-4">
        <div className="card-body">
          <h2 className="card-title">Change Requests</h2>
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
        </div>
      </div>
    </div>
  );
};

export default ATCInterface;