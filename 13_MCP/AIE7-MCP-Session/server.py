from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
import requests
import json
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

@mcp.tool()
def get_weather(location: str, include_forecast: bool = False) -> str:
    """Get current weather information for a specific location using Open-Meteo API.
    
    Args:
        location: City name, coordinates (lat,lon), or address
        include_forecast: If True, includes 7-day forecast data
    
    Returns:
        Weather information in a readable format
    """
    try:
        # First, get coordinates for the location using Open-Meteo's geocoding
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        geocoding_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        geocoding_response = requests.get(geocoding_url, params=geocoding_params)
        geocoding_response.raise_for_status()
        geocoding_data = geocoding_response.json()
        
        if not geocoding_data.get("results"):
            return f"Location '{location}' not found. Please try a different location name."
        
        result = geocoding_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        location_name = result["name"]
        country = result.get("country", "")
        
        # Get weather data
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weathercode,surface_pressure,wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }
        
        if include_forecast:
            weather_params["daily"] = "temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,wind_speed_10m_max"
            weather_params["forecast_days"] = 7
        
        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Format the response
        current = weather_data["current"]
        
        # Weather code interpretation
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        
        weather_description = weather_codes.get(current["weathercode"], "Unknown")
        
        result_text = f"Weather for {location_name}, {country}\n"
        result_text += f"Coordinates: {lat:.2f}, {lon:.2f}\n\n"
        result_text += f"Current Temperature: {current['temperature_2m']}°C\n"
        result_text += f"Feels Like: {current['apparent_temperature']}°C\n"
        result_text += f"Conditions: {weather_description}\n"
        result_text += f"Humidity: {current['relative_humidity_2m']}%\n"
        result_text += f"Precipitation: {current['precipitation']} mm\n"
        result_text += f"Wind: {current['wind_speed_10m']} km/h at {current['wind_direction_10m']}°\n"
        result_text += f"Pressure: {current['surface_pressure']} hPa\n"
        
        if include_forecast and "daily" in weather_data:
            result_text += "\n📅 7-Day Forecast:\n"
            daily = weather_data["daily"]
            for i in range(len(daily["time"])):
                date = daily["time"][i]
                max_temp = daily["temperature_2m_max"][i]
                min_temp = daily["temperature_2m_min"][i]
                weather_desc = weather_codes.get(daily["weathercode"][i], "Unknown")
                precipitation = daily["precipitation_sum"][i]
                wind_max = daily["wind_speed_10m_max"][i]
                
                result_text += f"  {date}: {min_temp}°C - {max_temp}°C, {weather_desc}"
                if precipitation > 0:
                    result_text += f", {precipitation}mm rain"
                result_text += f", max wind {wind_max} km/h\n"
        
        return result_text
        
    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")