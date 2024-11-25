import requests
import json
import re

def get_city_coordinates(city_name: str):
    """Fetches latitude and longitude for a given city using Nominatim API with User-Agent header."""
    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": city_name,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "just-agents/1.0 (maria@example.com)"
        }

        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        latitude = float(data[0]["lat"])
        longitude = float(data[0]["lon"])
        return {"latitude": latitude, "longitude": longitude}

    except requests.RequestException as e:
        print(f"Error fetching coordinates: {e}")
        return None

def get_weather_by_city(city: str):
    """Gets the current weather by a city name"""

    coordinates = get_city_coordinates(city)

    if not coordinates:
        return json.dumps({
            "error": f"Could not find coordinates for {city}."
        })

    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]

    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        current_weather = data.get("current_weather", {})
        temperature = current_weather.get("temperature", "N/A")
        windspeed = current_weather.get("windspeed", "N/A")
        description = f"Windspeed: {windspeed} km/h"

        result = json.dumps({
            "city": city,
            "temperature": str(temperature),
            "unit": "celsius",
            "description": description
        }, indent=2)
        return result

    except requests.RequestException as e:
        return json.dumps({
            "error": f"Could not fetch weather data: {str(e)}"
        })


"""
This example shows how a function can be used to call a function which potentially can have an external API call.
"""
def mock_get_current_weather(location: str):
    """Gets the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})