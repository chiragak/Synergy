# Flight Navigation System

[Project Documentation](https://drive.google.com/file/d/107TeQWKJnvwBJvJFm4SFlK1C0Obgn8Qp/view?usp=drivesdk)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
  - [Backend](#backend)
  - [Frontend](#frontend)
    - [Pilot Interface](#pilot-interface)
    - [ATC Interface](#atc-interface)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Flight Navigation System is designed to enhance flight navigation by minimizing human errors and leveraging automated navigation mechanisms. This system helps in optimal route planning and real-time risk assessment by considering various challenges such as adverse weather conditions, GPS signal unavailability, and electronic system failures. It includes interfaces for both pilots and air traffic controllers (ATC) to facilitate real-time decision-making.

## Features
- **Optimal Route Planning**: Automatically calculates the best flight paths considering weather, GPS signals, and other factors.
- **Real-Time Risk Assessment**: Provides real-time updates on potential risks and suggests alternative routes.
- **Pilot Interface**: Displays current flight information, weather conditions, and allows for requesting alternative routes.
- **ATC Interface**: Allows air traffic controllers to review and approve alternative route requests from pilots.
- **Database Initialization**: Scripts to initialize the database with relevant data including airlines, airplanes, airports, routes, and nearest airports.

## Technologies Used
- **Backend**: Flask, PyMongo, Pandas
- **Frontend**: React, Axios, Leaflet, Bootstrap
- **Database**: MongoDB

## Setup and Installation

### Backend

1. Navigate to the backend directory:
    ```bash
    cd backend
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the `backend` directory and add your MongoDB URI and OpenWeather API key.
    ```bash
    MONGO_URI="your_mongo_uri"
    OPENWEATHER_API_KEY="your_openweather_api_key"
    SECRET_KEY="your_secret_key"
    ```

5. Initialize the database:
    ```bash
    flask run
    ```

6. Open a new terminal and send a POST request to initialize the database:
    ```bash
    curl -X POST http://localhost:5000/api/initdb
    ```

### Frontend

#### Pilot Interface

1. Navigate to the pilot-interface directory:
    ```bash
    cd frontend/pilot-interface
    ```

2. Install the dependencies:
    ```bash
    npm install
    ```

3. Start the development server:
    ```bash
    npm start
    ```

#### ATC Interface

1. Navigate to the atc-interface directory:
    ```bash
    cd frontend/atc-interface
    ```

2. Install the dependencies:
    ```bash
    npm install
    ```

3. Start the development server:
    ```bash
    npm start
    ```

## Usage

1. Open the Pilot Interface in your browser at `http://localhost:3000`.
2. Open the ATC Interface in your browser at `http://localhost:3001`.
3. Use the Pilot Interface to view current flight information and request alternative routes.
4. Use the ATC Interface to review and approve alternative route requests.

## API Endpoints

- **GET /api/weather**: Get weather information based on latitude and longitude.
- **GET /api/route_by_airline**: Get route information based on airline ID, source airport ID, and destination airport ID.
- **POST /api/requestroute**: Request an alternative route.
- **GET /api/change-requests**: Get all change requests.
- **POST /api/approve-change-route**: Approve or reject a change request.
- **GET /api/all_routes**: Get all routes with pagination.
- **GET /api/nearest-airports**: Get nearest airports based on latitude and longitude.
- **GET /api/approved-change-requests**: Get approved change requests for a specific airline.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.



