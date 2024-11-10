from typing import List, Tuple, Optional
import requests
from math import radians, sin, cos, sqrt, atan2
import time
from bs4 import BeautifulSoup
import re
from io import StringIO
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
  """
  Calculate the haversine distance between two coordinate points
  """
  R = 6371  # Earth radius in kilometers

  lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  return R * c * 1000  # Convert to meters

def getNearest(lat: float, lon: float, radius: int = 10000, *, features: List = ["node", "way"], filters: List[List[Tuple[str, str]]], output_format="json",
    out_type="center"):

  """
    Search and retrieve nearest place within a given radius from a given latitude and longitude.

    :param lat: Latitude of point to search around from
    :param lon: Longitude of point to search around from
    :param radius: Radius of search in meters
    :param features: List of features to search for (default: ["node", "way"])
    :param filters: List of List of tag filters (e.g., [[("leisure", "park")], [("amenity", "restaurant"), ("cuisine", "italian")]] to get both nearby parks and italian resturants in one call)
    :param out_type: Output type (default: "center")
    :param output_format: Output format (default: "json")
    :return: Formatted place information string
    """

  overpass_url = "https://overpass-api.de/api/interpreter"

  # Query Creation
  overpass_query = f"[out:{output_format}];\n("
  for feature in features:
      for filter_group in filters:
          overpass_query += f"\n  {feature}"
          for key, value in filter_group:
              overpass_query += f"['{key}'='{value}']"
          overpass_query += f"(around:{radius},{lat},{lon});"
  overpass_query += f"\n);\nout {out_type};"

  overpass_params = {
      "data": overpass_query
  }
  overpass_headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-US,en;q=0.9',
      'Referer': 'https://overpass-api.de/'
  }

  overpass_response = requests.get(overpass_url, params=overpass_params, headers=overpass_headers)
  print("server response time:", overpass_response.elapsed.total_seconds())
  overpass_response.raise_for_status()  # Raise an exception for bad responses

  data = overpass_response.json()

  # print(json.dumps(data, indent=4))


  # Stores filters as keywords in category_names (i.e. [("leisure", "park")] as "park" or [("amenity", "restaurant"), ("cuisine", "italian")] as "resturant_italian")
  category_names = []

  for filter_group in filters:
    name = []
    for key, value in filter_group:
        name.append(value)
    category_names.append("_".join(name))

  categories = {name: [] for name in category_names}

  # Organize data by category
  for element in data['elements']:
    exit_flag = False
    for key, value in element["tags"].items():
      for categories_values in categories.keys():
        if categories_values == value:
          element_lat = element.get('lat') or element['center']['lat']
          element_lon = element.get('lon') or element['center']['lon']
          distance = haversine_distance(lat, lon, element_lat, element_lon)
          categories[value].append({"name": element['tags'].get('name', 'Unnamed'), "distance": distance, "lat": element_lat, "lon": element_lon})
          exit_flag = True # if element is appended, move on to the next element
      if exit_flag:
        break

  # dict of dict that holds the information of the closest place
  closest_places = {key: None for key in category_names}

  # find the shortest distance place from the property
  for category in categories.keys():
    if categories[category]:
      closest_places[category] = min(categories[category], key=lambda x: x["distance"])

  return closest_places

# helper function that removes tags (i.e. [("leisure", "park")] from filters given the word park as a keyword)
def remove_tags_by_value(filters, keyword):
  new_filter = []
  for filter_group in filters:
    keyword_exists = False
    for tag in filter_group:
      if keyword in tag:
        keyword_exists = True
    if not keyword_exists:
      new_filter.append(filter_group)
  return new_filter

def findNearestWithIncreasingRadius(lat: float, lon: float, initial_radius: int = 1000, max_radius: int = 50000, increment: int = 10000, filters: List[List[Tuple[str, str]]] = [[("leisure", "park")], [("highway", "motorway")], [("shop", "supermarket")], [("name", "Costco")]]):

  """
  Find the nearest place within a given radius from a given latitude and longitude.
  f the API doesn't return a place using the radius search, expand the radius and search again until found or reaches the max radius param.

  :param lat: Latitude of point to search around from
  :param lon: Longitude of point to search around from
  :param initial_radius: Initial radius of search in meters
  :param max_radius: Maximum radius of search in meters
  :param increment: Increment of radius in meters
  :param filters: List of List of tag filters (e.g., [[("leisure", "park")], [("amenity", "restaurant"), ("cuisine", "italian")]] to get both nearby parks and italian resturants in one call)
  :return: Dictionary of place information strings for each category provided in the filters parameter
  """

  current_radius = initial_radius
  current_filter = filters
  # ideal goal is to make result_dict the same as response_dict but not always possible with places with no close category
  result_dict = {}
  while current_radius <= max_radius and len(current_filter) > 0: # max out iterations or all categories requested are populated with values
    try:
      response_dict = getNearest(lat, lon, radius=current_radius, filters=current_filter)
      for category, place_info in response_dict.items():
        if place_info:
          result_dict[category] = place_info
          # update filter to not include categories that have information
          current_filter = remove_tags_by_value(current_filter, category)
          print(f"Nearest {category}: {place_info.get('name')} at coordinates ({place_info.get('lat')}, {place_info.get('lon')}), approximately {place_info.get('distance'):.2f} meters away")
        else:
          print(f"Cannot find {category} using coordinates ({lat}, {lon}), {current_radius} meters away. Retrying...")
          current_radius += increment  # Increase the radius
    except requests.exceptions.HTTPError as http_err:
      print(f"HTTP error occurred: {http_err}")  # e.g., 404 or 500
    except requests.exceptions.RequestException as err:
      print(f"Other error occurred: {err}")  # Handle other request-related errors
    time.sleep(1) # API documentation asks for minimum of 1 seconds between each request
  return result_dict

def get_california_city_wiki_url(city_name):
    def is_california_city_page(url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return False

            # Check if redirected to disambiguation
            if 'disambiguation' in response.url:
                return False

            soup = BeautifulSoup(response.content, 'html.parser')

            # Get the main content div
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                return False

            # Find first paragraph after content div
            # This skips the infobox and finds the actual first text paragraph
            first_p = None
            for p in soup.find_all('p'):
                # Skip if paragraph is inside a table
                if not p.find_parent('table'):
                    if p.text.strip():  # Check if paragraph has text
                        first_p = p
                        break

            # Check if page mentions its a city
            if first_p:
                text = first_p.text.lower()

                city_indicators = [
                    'city', 'town', 'municipality', 'metropolitan',
                    'county', 'population', 'census-designated place',
                    'incorporated', 'unincorporated', 'neighborhood'
                ]

                # Check for city indicators
                if not any(indicator in text for indicator in city_indicators):
                    return False

                # Check for geographic coordinates
                has_coords = bool(soup.find('span', class_='geo'))

                # Check infobox
                infobox = soup.find('table', class_=['infobox', 'geography', 'vcard'])
                if infobox:
                    infobox_text = infobox.text.lower()
                    has_city_info = any(x in infobox_text for x in [
                        'population', 'density', 'elevation', 'coordinates',
                        'mayor', 'timezone'
                    ])

                    if has_city_info:
                        return True
            return False

        except Exception as e:
            print(f"Error checking URL {url}: {str(e)}")
            return False

    try:
        # Clean city name
        clean_city = city_name.strip()
        if clean_city.lower().endswith(', california'):
            clean_city = clean_city[:-11].strip()
        elif clean_city.lower().endswith(' ca'):
            clean_city = clean_city[:-3].strip()
        clean_city = clean_city.title()
        variants = [
            f"{clean_city.replace(' ', '_')},_California",  # City,_California
            clean_city.replace(' ', '_'),  # Just city name
            f"{clean_city.replace(' ', '_')}_city",  # City_city
            f"{clean_city.replace(' ', '_')}_(California)",  # City_(California)
            f"{clean_city.replace(' ', '_')},_CA"  # City,_CA
            f"{clean_city.replace(' ', '_')},_Los_Angeles"  # City,_CA
        ]

        # Try each variant
        for variant in variants:
            url = f'https://en.wikipedia.org/wiki/{variant}'
            if is_california_city_page(url):
                return url

        return None

    except Exception as e:
        print(f"Error processing {city_name}: {str(e)}")
        return None
    
# Helper function to extract numeric values from a string
def extract_numeric_value(value_str):
    """
    Extracts numeric values from a string, handling commas, negative signs, and various formats.
    Returns None if no valid number is found.
    """
    if value_str:
        # Use regular expression to find the numeric part, including decimals and negative numbers
        match = re.findall(r"[-+]?\d*\.\d+|\d+", value_str.replace(',', '').replace('−', '-'))
        if match:
            return float(match[0])
    return None

# Function to scrape climate data (average temperature, rainfall, elevation, population density) for a given city
def get_city_climate_data(city_name):
    # Format city name for URL
    url = get_california_city_wiki_url(city_name)

    time.sleep(1)

    # Send request to Wikipedia
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Initialize dictionary to store climate and population data
    climate_data = {
        'city': city_name,
        'elevation': None,
        'rainfall': None,
        'mean_daily_high_temperature': None,
        'mean_daily_low_temperature': None,
        'population_density': None,
        'latitude': None,
        'longitude': None
    }

    # Scrape Elevation and Population Density (usually in the infobox on the right-hand side)
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        for row in infobox.find_all('tr'):
            if 'Elevation' in row.text:
                elevation_data = row.find_next('td').text.strip()
                climate_data['elevation'] = extract_numeric_value(elevation_data)

            # Scrape population density if available
            if 'Population density' in row.text or 'Density' in row.text:
                pop_density_data = row.find_next('td').text.strip()
                climate_data['population_density'] = extract_numeric_value(pop_density_data)

            # Scrape Coordinate Values of Cities
            if 'Coordinates' in row.text:
                # Extract the coordinates
                coord_link = row.find('span', {'class': 'geo'})
                if coord_link:
                    coords = coord_link.text.split('; ')
                    latitude = float(coords[0])
                    longitude = float(coords[1])
                    climate_data['latitude'] = latitude
                    climate_data['longitude'] = longitude

    # Scrape Temperature and Rainfall from climate table
    climate_table = None
    tables = soup.find_all('table', class_='wikitable')

    for table in tables:
        if 'climate' in str(table).lower() or 'weather' in str(table).lower():
            climate_table = table
            break

    if climate_table:
        df = pd.read_html(StringIO(str(climate_table)))[0]

        # If there's a multi-index in columns, flatten it
        # Basically ignore multi-index column
        if isinstance(df.columns, pd.MultiIndex):
            # Join multi-level column names with underscore
            df.columns = [f"{col[1]}" if pd.notna(col[1]) else col[0]
                        for col in df.columns]

            # Remove any double underscores and strip whitespace
            df.columns = [col.replace('__', '_').strip()
                        for col in df.columns]

        # Helper function to locate mean value for the year for a specified row in climate dataframe
        def tableFinder(df: pd.DataFrame, text_search: str) -> pd.Series:
            return df.loc[df["Month"].str.contains(text_search, case=False), "Year"]

        # Helper function to convert Series of strings into dict of floats for temperatures
        def cleanTableValues(series: pd.Series) -> List[float]:
            if not series.empty:
                values = series.values[0].split()
                values[1] = re.sub(r'()', '', values[0]) # remove parenthesis from Celcius value
                values = list(map(float, values))
                return values
            else:
                return [None, None]

        mean_daily_max = tableFinder(df, "mean daily max")
        mean_daily_max = cleanTableValues(mean_daily_max)
        mean_daily_max = {"C": mean_daily_max[0], "F": mean_daily_max[1]}
        climate_data["mean_daily_high_temperature"] = mean_daily_max.get("F")

        mean_daily_min = tableFinder(df, "mean daily min")
        mean_daily_min = cleanTableValues(mean_daily_min)
        mean_daily_min = {"C": mean_daily_min[0], "F": mean_daily_min[1]}
        climate_data["mean_daily_low_temperature"] = mean_daily_min.get("F")

        mean_precipiation = tableFinder(df, "Average precipitation inches")
        mean_precipiation = cleanTableValues(mean_precipiation)
        mean_precipiation = {"mm": mean_precipiation[0], "inches": mean_precipiation[1]}
        climate_data["rainfall"] = mean_precipiation.get("inches")

    return climate_data

def geocode_address(address: str, city: str, zipcode: int, state: str = "California") -> Optional[Tuple[float, float]]:
    """
    Geocode an address in California using OpenStreetMap's Nominatim API.
    
    Args:
        address (str): The address to geocode
        state (str): The state to search in (defaults to California)
    
    Returns:
        Optional[Tuple[float, float]]: A tuple of (latitude, longitude) if found, None if not found
    """
    # Create proper headers with user agent (required by Nominatim)
    headers = {
        'User-Agent': 'AddressGeocoder/1.0',
        'Accept': 'application/json'
    }
    
    # Base URL for Nominatim API
    base_url = "https://nominatim.openstreetmap.org/search"
    
    # Parameters for the API request
    params = {
        'street': address,
        'city': city,
        'state': state,
        'zipcode': zipcode,
        'country': 'USA',
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        # Parse the response
        results = response.json()
        
        # Respect Nominatim's usage policy with a 1 second delay
        time.sleep(1)
        
        if not results:
            print(f"No results found for address: {address}")
            return None
        
        # Extract coordinates from the first result
        location = results[0]
        latitude = float(location['lat'])
        longitude = float(location['lon'])
        
        return (latitude, longitude)
    
    except requests.RequestException as e:
        print(f"Error occurred while geocoding address: {str(e)}")
        return None
    
    except (KeyError, ValueError) as e:
        print(f"Error parsing response data: {str(e)}")
        return None