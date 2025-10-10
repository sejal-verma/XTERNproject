# Weather Data Ingestion System
# Implementation for Task 3: Weather Data Adapters

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class WeatherFeatures:
    """Standardized weather feature data structure"""
    cell_id: str
    horizon_h: int
    timestamp: datetime
    temp_2m: float          # Temperature at 2m (°F)
    heat_index: float       # Heat index (°F)
    wind_speed: float       # Sustained wind speed (mph)
    wind_gust: float        # Wind gust speed (mph)
    precip_rate: float      # Precipitation rate (mm/h)
    snow_rate: float        # Snow rate (cm/h)
    dewpoint: float         # Dewpoint temperature (°F)
    relative_humidity: float # Relative humidity (%)
    storm_probability: float # Storm probability [0,1]
    confidence: float       # Data confidence [0,1]


class WeatherAdapter(ABC):
    """Abstract base class for weather data adapters"""
    
    def __init__(self, config: Dict, cache_dir: str = "data/raw"):
        self.config = config
        self.cache_dir = cache_dir
        self.session = self._setup_session()
        self._ensure_cache_dir()
    
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy and rate limiting"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config['runtime']['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout
        session.timeout = self.config['runtime']['api_timeout']
        
        return session
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @abstractmethod
    def fetch_forecast(self, horizon_h: int, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Fetch and aggregate weather data for specified horizon
        
        Args:
            horizon_h: Forecast horizon in hours (12, 24, 36, 48)
            grid: GeoDataFrame with hexagonal grid cells
            
        Returns:
            DataFrame with standardized weather features
        """
        pass
    
    @abstractmethod
    def get_available_parameters(self) -> List[str]:
        """Return list of available weather parameters"""
        pass    

    def _cache_response(self, data: Dict, cache_key: str) -> None:
        """Cache raw API response"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.debug(f"Cached response to: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to cache response: {e}")
    
    def _load_cached_response(self, cache_key: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Load cached API response if still fresh"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check age
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age > max_age_hours * 3600:
            logging.debug(f"Cache expired: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logging.debug(f"Loaded cached response: {cache_file}")
            return data
        except Exception as e:
            logging.warning(f"Failed to load cached response: {e}")
            return None
    
    def _spatial_aggregate_to_grid(self, weather_points: gpd.GeoDataFrame, 
                                  grid: gpd.GeoDataFrame,
                                  value_columns: List[str],
                                  agg_method: str = "mean") -> pd.DataFrame:
        """Aggregate weather point data to grid cells
        
        Args:
            weather_points: GeoDataFrame with weather data points
            grid: GeoDataFrame with hexagonal grid
            value_columns: List of columns to aggregate
            agg_method: Aggregation method ('mean', 'max', 'min')
            
        Returns:
            DataFrame with aggregated weather data by cell_id
        """
        try:
            # Ensure same CRS
            if weather_points.crs != grid.crs:
                weather_points = weather_points.to_crs(grid.crs)
            
            # Spatial join - find which grid cell each weather point belongs to
            joined = gpd.sjoin(weather_points, grid[['cell_id', 'geometry']], 
                             how='inner', predicate='intersects')
            
            # Aggregate by cell_id
            agg_funcs = {col: agg_method for col in value_columns}
            aggregated = joined.groupby('cell_id').agg(agg_funcs).reset_index()
            
            # Flatten column names if needed
            if isinstance(aggregated.columns, pd.MultiIndex):
                aggregated.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
                                    for col in aggregated.columns]
            
            return aggregated
            
        except Exception as e:
            logging.error(f"Spatial aggregation failed: {e}")
            raise
    
    def validate_weather_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate weather data quality and completeness
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['cell_id', 'horizon_h', 'temp_2m', 'wind_speed', 'precip_rate']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for null values in critical columns
        for col in required_cols:
            if col in data.columns and data[col].isna().any():
                null_count = data[col].isna().sum()
                issues.append(f"Null values in {col}: {null_count}/{len(data)}")
        
        # Check value ranges
        if 'temp_2m' in data.columns:
            temp_range = data['temp_2m'].describe()
            if temp_range['min'] < -50 or temp_range['max'] > 150:  # °F
                issues.append(f"Temperature out of range: {temp_range['min']:.1f} to {temp_range['max']:.1f}°F")
        
        if 'wind_speed' in data.columns:
            wind_range = data['wind_speed'].describe()
            if wind_range['min'] < 0 or wind_range['max'] > 200:  # mph
                issues.append(f"Wind speed out of range: {wind_range['min']:.1f} to {wind_range['max']:.1f} mph")
        
        if 'precip_rate' in data.columns:
            precip_range = data['precip_rate'].describe()
            if precip_range['min'] < 0 or precip_range['max'] > 100:  # mm/h
                issues.append(f"Precipitation rate out of range: {precip_range['min']:.1f} to {precip_range['max']:.1f} mm/h")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class NOAAAdapter(WeatherAdapter):
    """NOAA/NWS gridpoint forecast API adapter"""
    
    def __init__(self, config: Dict, cache_dir: str = "data/raw"):
        super().__init__(config, cache_dir)
        self.base_url = "https://api.weather.gov"
        self.user_agent = "MISO-Weather-Stress-Heatmap/1.0"
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def get_available_parameters(self) -> List[str]:
        """Return list of available NOAA weather parameters"""
        return [
            'temperature', 'dewpoint', 'maxTemperature', 'minTemperature',
            'relativeHumidity', 'windSpeed', 'windGust', 'windDirection',
            'skyCover', 'weather', 'hazards', 'probabilityOfPrecipitation',
            'quantitativePrecipitation', 'iceAccumulation', 'snowfallAmount'
        ]   
 
    def fetch_forecast(self, horizon_h: int, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Fetch NOAA forecast data for grid cells
        
        Args:
            horizon_h: Forecast horizon in hours
            grid: GeoDataFrame with hexagonal grid cells
            
        Returns:
            DataFrame with weather features for each grid cell
        """
        try:
            logging.info(f"Fetching NOAA forecast data for {horizon_h}h horizon...")
            
            # Generate cache key
            cache_key = f"noaa_forecast_{horizon_h}h_{datetime.now().strftime('%Y%m%d_%H')}"
            
            # Try to load from cache first
            cached_data = self._load_cached_response(cache_key)
            if cached_data is not None:
                logging.info("Using cached NOAA data")
                return pd.DataFrame(cached_data)
            
            # Fetch fresh data
            weather_data = []
            target_time = datetime.now() + timedelta(hours=horizon_h)
            
            # Sample a subset of grid cells for demo (to avoid rate limiting)
            sample_size = min(50, len(grid))  # Limit to 50 cells for demo
            grid_sample = grid.sample(n=sample_size, random_state=42)
            
            for idx, cell in grid_sample.iterrows():
                try:
                    # Get forecast for this cell's centroid
                    lat, lon = cell['centroid_lat'], cell['centroid_lon']
                    cell_weather = self._fetch_gridpoint_forecast(lat, lon, target_time)
                    
                    if cell_weather:
                        cell_weather['cell_id'] = cell['cell_id']
                        cell_weather['horizon_h'] = horizon_h
                        weather_data.append(cell_weather)
                    
                    # Rate limiting
                    time.sleep(0.1)  # 10 requests per second max
                    
                except Exception as e:
                    logging.warning(f"Failed to fetch weather for cell {cell['cell_id']}: {e}")
                    continue
            
            if not weather_data:
                raise ValueError("No weather data retrieved from NOAA API")
            
            # Convert to DataFrame
            df = pd.DataFrame(weather_data)
            
            # Cache the results
            self._cache_response(df.to_dict('records'), cache_key)
            
            logging.info(f"Retrieved NOAA weather data for {len(df)} cells")
            return df
            
        except Exception as e:
            logging.error(f"NOAA forecast fetch failed: {e}")
            raise
    
    def _fetch_gridpoint_forecast(self, lat: float, lon: float, 
                                 target_time: datetime) -> Optional[Dict]:
        """Fetch forecast for a specific gridpoint"""
        try:
            # Step 1: Get gridpoint info
            points_url = f"{self.base_url}/points/{lat:.4f},{lon:.4f}"
            response = self.session.get(points_url)
            
            if response.status_code != 200:
                logging.debug(f"Points API failed for {lat},{lon}: {response.status_code}")
                return None
            
            points_data = response.json()
            
            # Step 2: Get forecast from gridpoint
            forecast_url = points_data['properties']['forecastGridData']
            forecast_response = self.session.get(forecast_url)
            
            if forecast_response.status_code != 200:
                logging.debug(f"Forecast API failed: {forecast_response.status_code}")
                return None
            
            forecast_data = forecast_response.json()
            
            # Step 3: Extract weather parameters for target time
            weather_params = self._extract_weather_parameters(forecast_data, target_time)
            
            return weather_params
            
        except Exception as e:
            logging.debug(f"Gridpoint forecast failed for {lat},{lon}: {e}")
            return None 
   
    def _extract_weather_parameters(self, forecast_data: Dict, 
                                   target_time: datetime) -> Dict:
        """Extract weather parameters from NOAA forecast response"""
        try:
            properties = forecast_data['properties']
            
            # Initialize result with defaults
            result = {
                'timestamp': target_time,
                'temp_2m': np.nan,
                'heat_index': np.nan,
                'wind_speed': np.nan,
                'wind_gust': np.nan,
                'precip_rate': np.nan,
                'snow_rate': np.nan,
                'dewpoint': np.nan,
                'relative_humidity': np.nan,
                'storm_probability': np.nan,
                'confidence': 0.8  # Base confidence for NOAA data
            }
            
            # Extract temperature (convert K to F)
            if 'temperature' in properties:
                temp_data = properties['temperature']
                temp_value = self._interpolate_time_series(temp_data, target_time)
                if temp_value is not None:
                    result['temp_2m'] = (temp_value - 273.15) * 9/5 + 32  # K to F
            
            # Extract dewpoint (convert K to F)
            if 'dewpoint' in properties:
                dewpoint_data = properties['dewpoint']
                dewpoint_value = self._interpolate_time_series(dewpoint_data, target_time)
                if dewpoint_value is not None:
                    result['dewpoint'] = (dewpoint_value - 273.15) * 9/5 + 32  # K to F
            
            # Calculate heat index if we have temp and dewpoint
            if not np.isnan(result['temp_2m']) and not np.isnan(result['dewpoint']):
                result['heat_index'] = self._calculate_heat_index(
                    result['temp_2m'], result['dewpoint']
                )
            
            # Extract relative humidity
            if 'relativeHumidity' in properties:
                rh_data = properties['relativeHumidity']
                rh_value = self._interpolate_time_series(rh_data, target_time)
                if rh_value is not None:
                    result['relative_humidity'] = rh_value
            
            # Extract wind speed (convert m/s to mph)
            if 'windSpeed' in properties:
                wind_data = properties['windSpeed']
                wind_value = self._interpolate_time_series(wind_data, target_time)
                if wind_value is not None:
                    result['wind_speed'] = wind_value * 2.237  # m/s to mph
            
            # Extract wind gust (convert m/s to mph)
            if 'windGust' in properties:
                gust_data = properties['windGust']
                gust_value = self._interpolate_time_series(gust_data, target_time)
                if gust_value is not None:
                    result['wind_gust'] = gust_value * 2.237  # m/s to mph
                else:
                    # Estimate gust as 1.3x sustained wind if not available
                    if not np.isnan(result['wind_speed']):
                        result['wind_gust'] = result['wind_speed'] * 1.3
            
            # Extract precipitation (convert kg/m²/s to mm/h)
            if 'quantitativePrecipitation' in properties:
                precip_data = properties['quantitativePrecipitation']
                precip_value = self._interpolate_time_series(precip_data, target_time)
                if precip_value is not None:
                    result['precip_rate'] = precip_value * 3600  # kg/m²/s to mm/h
            
            # Extract snowfall (convert m to cm/h, assuming 1-hour accumulation)
            if 'snowfallAmount' in properties:
                snow_data = properties['snowfallAmount']
                snow_value = self._interpolate_time_series(snow_data, target_time)
                if snow_value is not None:
                    result['snow_rate'] = snow_value * 100  # m to cm
            
            # Calculate storm probability
            result['storm_probability'] = self._calculate_storm_probability(
                result['precip_rate'], result['wind_gust']
            )
            
            return result
            
        except Exception as e:
            logging.debug(f"Parameter extraction failed: {e}")
            return {}
    
    def _interpolate_time_series(self, time_series_data: Dict, 
                                target_time: datetime) -> Optional[float]:
        """Interpolate value from NOAA time series data"""
        try:
            if 'values' not in time_series_data:
                return None
            
            values = time_series_data['values']
            if not values:
                return None
            
            # Convert target time to ISO string for comparison
            target_iso = target_time.isoformat()
            
            # Find closest time points
            times = []
            vals = []
            
            for item in values:
                if 'validTime' in item and 'value' in item:
                    # Parse time range (e.g., "2023-10-10T12:00:00+00:00/PT1H")
                    time_str = item['validTime'].split('/')[0]
                    times.append(time_str)
                    vals.append(item['value'])
            
            if not times:
                return None
            
            # Simple approach: find closest time
            closest_idx = 0
            min_diff = float('inf')
            
            for i, time_str in enumerate(times):
                try:
                    time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    diff = abs((time_obj - target_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                except:
                    continue
            
            return vals[closest_idx] if vals else None
            
        except Exception as e:
            logging.debug(f"Time series interpolation failed: {e}")
            return None
    
    def _calculate_heat_index(self, temp_f: float, dewpoint_f: float) -> float:
        """Calculate heat index from temperature and dewpoint"""
        try:
            # Calculate relative humidity from temp and dewpoint
            # Simplified approximation
            rh = 100 * np.exp((17.625 * (dewpoint_f - 32) * 5/9) / 
                             (243.04 + (dewpoint_f - 32) * 5/9)) / \
                      np.exp((17.625 * (temp_f - 32) * 5/9) / 
                             (243.04 + (temp_f - 32) * 5/9))
            
            # Heat index calculation (Rothfusz equation)
            if temp_f < 80:
                return temp_f  # No heat index below 80°F
            
            hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh 
                  - 0.22475541 * temp_f * rh - 6.83783e-3 * temp_f**2 
                  - 5.481717e-2 * rh**2 + 1.22874e-3 * temp_f**2 * rh 
                  + 8.5282e-4 * temp_f * rh**2 - 1.99e-6 * temp_f**2 * rh**2)
            
            return max(temp_f, hi)  # Heat index should not be less than temperature
            
        except:
            return temp_f  # Fallback to temperature
    
    def _calculate_storm_probability(self, precip_rate: float, 
                                   wind_gust: float) -> float:
        """Calculate storm probability from precipitation and wind"""
        try:
            if np.isnan(precip_rate) or np.isnan(wind_gust):
                return 0.0
            
            # Storm conditions: precipitation > 0 AND wind gust >= 35 mph
            if precip_rate > 0 and wind_gust >= 35:
                return 1.0
            
            # Partial storm conditions based on intensity
            storm_score = 0.0
            
            # Precipitation contribution (0-5 mm/h = 0-0.5, >5 mm/h = 0.5-1.0)
            if precip_rate > 0:
                storm_score += min(0.5, precip_rate / 10.0)
            
            # Wind contribution (20-35 mph = 0-0.5, >35 mph = 0.5-1.0)
            if wind_gust > 20:
                storm_score += min(0.5, (wind_gust - 20) / 30.0)
            
            return min(1.0, storm_score)
            
        except:
            return 0.0


class OpenMeteoAdapter(WeatherAdapter):
    """Open-Meteo API adapter as fallback for NOAA"""
    
    def __init__(self, config: Dict, cache_dir: str = "data/raw"):
        super().__init__(config, cache_dir)
        self.base_url = "https://api.open-meteo.com/v1/forecast"
    
    def get_available_parameters(self) -> List[str]:
        """Return list of available Open-Meteo parameters"""
        return [
            'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
            'apparent_temperature', 'precipitation', 'rain', 'snowfall',
            'windspeed_10m', 'windgusts_10m', 'winddirection_10m',
            'cloudcover', 'visibility', 'weathercode'
        ]
    
    def fetch_forecast(self, horizon_h: int, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Fetch Open-Meteo forecast data for grid cells"""
        try:
            logging.info(f"Fetching Open-Meteo forecast data for {horizon_h}h horizon...")
            
            # Generate cache key
            cache_key = f"openmeteo_forecast_{horizon_h}h_{datetime.now().strftime('%Y%m%d_%H')}"
            
            # Try to load from cache first
            cached_data = self._load_cached_response(cache_key)
            if cached_data is not None:
                logging.info("Using cached Open-Meteo data")
                return pd.DataFrame(cached_data)
            
            # Fetch fresh data
            weather_data = []
            
            # Sample a subset of grid cells for demo
            sample_size = min(50, len(grid))
            grid_sample = grid.sample(n=sample_size, random_state=42)
            
            # Batch requests by location (Open-Meteo supports multiple locations)
            batch_size = 10  # Process in batches to avoid URL length limits
            
            for i in range(0, len(grid_sample), batch_size):
                batch = grid_sample.iloc[i:i+batch_size]
                
                try:
                    batch_weather = self._fetch_batch_forecast(batch, horizon_h)
                    weather_data.extend(batch_weather)
                    
                    # Rate limiting
                    time.sleep(0.5)  # Be conservative with rate limiting
                    
                except Exception as e:
                    logging.warning(f"Failed to fetch weather batch {i//batch_size + 1}: {e}")
                    continue
            
            if not weather_data:
                raise ValueError("No weather data retrieved from Open-Meteo API")
            
            # Convert to DataFrame
            df = pd.DataFrame(weather_data)
            
            # Cache the results
            self._cache_response(df.to_dict('records'), cache_key)
            
            logging.info(f"Retrieved Open-Meteo weather data for {len(df)} cells")
            return df
            
        except Exception as e:
            logging.error(f"Open-Meteo forecast fetch failed: {e}")
            raise    

    def _fetch_batch_forecast(self, grid_batch: gpd.GeoDataFrame, 
                             horizon_h: int) -> List[Dict]:
        """Fetch forecast for a batch of grid cells"""
        try:
            # Prepare coordinates
            latitudes = grid_batch['centroid_lat'].tolist()
            longitudes = grid_batch['centroid_lon'].tolist()
            cell_ids = grid_batch['cell_id'].tolist()
            
            # Build API request
            params = {
                'latitude': latitudes,
                'longitude': longitudes,
                'hourly': [
                    'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                    'apparent_temperature', 'precipitation', 'rain', 'snowfall',
                    'windspeed_10m', 'windgusts_10m', 'cloudcover'
                ],
                'temperature_unit': 'fahrenheit',
                'windspeed_unit': 'mph',
                'precipitation_unit': 'mm',
                'timezone': 'UTC',
                'forecast_days': 3  # Cover up to 48h horizon
            }
            
            # Make API request
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract weather data for each location
            weather_data = []
            
            for i, cell_id in enumerate(cell_ids):
                try:
                    cell_weather = self._extract_openmeteo_parameters(
                        data, i, cell_id, horizon_h
                    )
                    if cell_weather:
                        weather_data.append(cell_weather)
                except Exception as e:
                    logging.debug(f"Failed to extract weather for cell {cell_id}: {e}")
                    continue
            
            return weather_data
            
        except Exception as e:
            logging.debug(f"Batch forecast failed: {e}")
            return []
    
    def _extract_openmeteo_parameters(self, data: Dict, location_idx: int,
                                     cell_id: str, horizon_h: int) -> Optional[Dict]:
        """Extract weather parameters from Open-Meteo response"""
        try:
            # Handle both single location and multi-location responses
            if isinstance(data, list):
                location_data = data[location_idx]
            else:
                # Multi-location response format
                location_data = data
            
            hourly = location_data['hourly']
            times = hourly['time']
            
            # Find the target time index (closest to horizon_h from now)
            target_time = datetime.now() + timedelta(hours=horizon_h)
            target_hour = target_time.strftime('%Y-%m-%dT%H:00')
            
            # Find closest time index
            time_idx = None
            for i, time_str in enumerate(times):
                if time_str.startswith(target_hour[:13]):  # Match YYYY-MM-DDTHH
                    time_idx = i
                    break
            
            if time_idx is None:
                # Fallback to closest available time
                time_idx = min(len(times) - 1, horizon_h)
            
            # Extract parameters
            result = {
                'cell_id': cell_id,
                'horizon_h': horizon_h,
                'timestamp': target_time,
                'confidence': 0.7  # Slightly lower confidence than NOAA
            }
            
            # Temperature
            if 'temperature_2m' in hourly:
                result['temp_2m'] = hourly['temperature_2m'][time_idx]
            
            # Heat index (use apparent temperature if available)
            if 'apparent_temperature' in hourly:
                result['heat_index'] = hourly['apparent_temperature'][time_idx]
            elif 'temp_2m' in result:
                result['heat_index'] = result['temp_2m']  # Fallback
            
            # Dewpoint
            if 'dewpoint_2m' in hourly:
                result['dewpoint'] = hourly['dewpoint_2m'][time_idx]
            
            # Relative humidity
            if 'relativehumidity_2m' in hourly:
                result['relative_humidity'] = hourly['relativehumidity_2m'][time_idx]
            
            # Wind speed
            if 'windspeed_10m' in hourly:
                result['wind_speed'] = hourly['windspeed_10m'][time_idx]
            
            # Wind gust
            if 'windgusts_10m' in hourly:
                result['wind_gust'] = hourly['windgusts_10m'][time_idx]
            elif 'wind_speed' in result:
                # Estimate gust as 1.3x sustained wind
                result['wind_gust'] = result['wind_speed'] * 1.3
            
            # Precipitation rate
            if 'precipitation' in hourly:
                result['precip_rate'] = hourly['precipitation'][time_idx]
            elif 'rain' in hourly:
                result['precip_rate'] = hourly['rain'][time_idx]
            
            # Snow rate (convert from mm to cm)
            if 'snowfall' in hourly:
                result['snow_rate'] = hourly['snowfall'][time_idx] / 10.0  # mm to cm
            
            # Calculate storm probability
            result['storm_probability'] = self._calculate_storm_probability(
                result.get('precip_rate', 0), result.get('wind_gust', 0)
            )
            
            # Fill missing values with defaults
            defaults = {
                'temp_2m': np.nan, 'heat_index': np.nan, 'wind_speed': 0.0,
                'wind_gust': 0.0, 'precip_rate': 0.0, 'snow_rate': 0.0,
                'dewpoint': np.nan, 'relative_humidity': np.nan,
                'storm_probability': 0.0
            }
            
            for key, default_val in defaults.items():
                if key not in result:
                    result[key] = default_val
            
            return result
            
        except Exception as e:
            logging.debug(f"Open-Meteo parameter extraction failed: {e}")
            return None
    
    def _calculate_storm_probability(self, precip_rate: float, 
                                   wind_gust: float) -> float:
        """Calculate storm probability (same logic as NOAA adapter)"""
        try:
            if np.isnan(precip_rate) or np.isnan(wind_gust):
                return 0.0
            
            # Storm conditions: precipitation > 0 AND wind gust >= 35 mph
            if precip_rate > 0 and wind_gust >= 35:
                return 1.0
            
            # Partial storm conditions
            storm_score = 0.0
            
            if precip_rate > 0:
                storm_score += min(0.5, precip_rate / 10.0)
            
            if wind_gust > 20:
                storm_score += min(0.5, (wind_gust - 20) / 30.0)
            
            return min(1.0, storm_score)
            
        except:
            return 0.0


class WeatherDataManager:
    """Main weather data management class with fallback logic"""
    
    def __init__(self, config: Dict, cache_dir: str = "data/raw"):
        self.config = config
        self.cache_dir = cache_dir
        
        # Initialize adapters
        self.primary_adapter = NOAAAdapter(config, cache_dir)
        self.fallback_adapter = OpenMeteoAdapter(config, cache_dir)
        
        # Track adapter usage
        self.adapter_usage = {'noaa': 0, 'openmeteo': 0, 'failures': 0}
    
    def fetch_weather_data(self, horizons: List[int], 
                          grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Fetch weather data for multiple horizons with automatic fallback
        
        Args:
            horizons: List of forecast horizons in hours [12, 24, 36, 48]
            grid: GeoDataFrame with hexagonal grid cells
            
        Returns:
            Combined DataFrame with weather data for all horizons
        """
        try:
            logging.info(f"Fetching weather data for horizons: {horizons}")
            
            all_weather_data = []
            
            for horizon_h in horizons:
                try:
                    # Try primary adapter (NOAA) first
                    logging.info(f"Attempting NOAA fetch for {horizon_h}h horizon...")
                    weather_df = self.primary_adapter.fetch_forecast(horizon_h, grid)
                    
                    # Validate data quality
                    is_valid, issues = self.primary_adapter.validate_weather_data(weather_df)
                    
                    if is_valid:
                        all_weather_data.append(weather_df)
                        self.adapter_usage['noaa'] += 1
                        logging.info(f"✓ NOAA data successful for {horizon_h}h")
                    else:
                        logging.warning(f"NOAA data quality issues: {issues}")
                        raise ValueError("Data quality validation failed")
                
                except Exception as e:
                    logging.warning(f"NOAA adapter failed for {horizon_h}h: {e}")
                    
                    try:
                        # Fallback to Open-Meteo
                        logging.info(f"Falling back to Open-Meteo for {horizon_h}h...")
                        weather_df = self.fallback_adapter.fetch_forecast(horizon_h, grid)
                        
                        # Validate fallback data
                        is_valid, issues = self.fallback_adapter.validate_weather_data(weather_df)
                        
                        if is_valid:
                            all_weather_data.append(weather_df)
                            self.adapter_usage['openmeteo'] += 1
                            logging.info(f"✓ Open-Meteo fallback successful for {horizon_h}h")
                        else:
                            logging.error(f"Fallback data quality issues: {issues}")
                            raise ValueError("Fallback data validation failed")
                    
                    except Exception as fallback_error:
                        logging.error(f"Both adapters failed for {horizon_h}h: {fallback_error}")
                        self.adapter_usage['failures'] += 1
                        
                        # Generate demo data as last resort
                        demo_data = self._generate_demo_weather_data(horizon_h, grid)
                        all_weather_data.append(demo_data)
                        logging.warning(f"Using demo weather data for {horizon_h}h")
            
            if not all_weather_data:
                raise ValueError("No weather data could be retrieved for any horizon")
            
            # Combine all horizons
            combined_df = pd.concat(all_weather_data, ignore_index=True)
            
            # Log summary
            self._log_weather_summary(combined_df)
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Weather data fetch failed: {e}")
            raise
    
    def _generate_demo_weather_data(self, horizon_h: int, 
                                   grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate synthetic demo weather data as fallback"""
        try:
            logging.info(f"Generating demo weather data for {horizon_h}h...")
            
            np.random.seed(42 + horizon_h)  # Reproducible but varied by horizon
            
            demo_data = []
            
            # Sample subset of grid for demo
            sample_size = min(50, len(grid))
            grid_sample = grid.sample(n=sample_size, random_state=42)
            
            for _, cell in grid_sample.iterrows():
                # Generate realistic weather values based on location
                lat = cell['centroid_lat']
                
                # Temperature varies by latitude and season
                base_temp = 70 - (lat - 40) * 2  # Cooler further north
                temp_variation = np.random.normal(0, 10)
                temp_2m = base_temp + temp_variation
                
                # Heat index slightly higher than temperature
                heat_index = temp_2m + np.random.uniform(0, 5)
                
                # Wind speeds
                wind_speed = np.random.exponential(8)  # Exponential distribution
                wind_gust = wind_speed * np.random.uniform(1.2, 1.8)
                
                # Precipitation (mostly zero with occasional events)
                precip_rate = 0.0
                if np.random.random() < 0.2:  # 20% chance of precipitation
                    precip_rate = np.random.exponential(2)
                
                # Snow (only if temperature is low)
                snow_rate = 0.0
                if temp_2m < 35 and np.random.random() < 0.1:
                    snow_rate = np.random.exponential(1)
                
                # Dewpoint and humidity
                dewpoint = temp_2m - np.random.uniform(5, 20)
                relative_humidity = np.random.uniform(30, 90)
                
                # Storm probability
                storm_prob = 0.0
                if precip_rate > 0 and wind_gust > 35:
                    storm_prob = 1.0
                elif precip_rate > 0 or wind_gust > 20:
                    storm_prob = min(1.0, (precip_rate / 10 + (wind_gust - 20) / 30) / 2)
                
                demo_data.append({
                    'cell_id': cell['cell_id'],
                    'horizon_h': horizon_h,
                    'timestamp': datetime.now() + timedelta(hours=horizon_h),
                    'temp_2m': temp_2m,
                    'heat_index': heat_index,
                    'wind_speed': wind_speed,
                    'wind_gust': wind_gust,
                    'precip_rate': precip_rate,
                    'snow_rate': snow_rate,
                    'dewpoint': dewpoint,
                    'relative_humidity': relative_humidity,
                    'storm_probability': storm_prob,
                    'confidence': 0.5  # Lower confidence for demo data
                })
            
            return pd.DataFrame(demo_data)
            
        except Exception as e:
            logging.error(f"Demo weather data generation failed: {e}")
            raise
    
    def _log_weather_summary(self, weather_df: pd.DataFrame) -> None:
        """Log summary of weather data retrieval"""
        logging.info("=== Weather Data Summary ===")
        logging.info(f"Total weather records: {len(weather_df):,}")
        logging.info(f"Unique cells: {weather_df['cell_id'].nunique()}")
        logging.info(f"Horizons covered: {sorted(weather_df['horizon_h'].unique())}")
        logging.info(f"Adapter usage: NOAA={self.adapter_usage['noaa']}, "
                    f"Open-Meteo={self.adapter_usage['openmeteo']}, "
                    f"Failures={self.adapter_usage['failures']}")
        
        # Data quality summary
        for col in ['temp_2m', 'wind_speed', 'precip_rate']:
            if col in weather_df.columns:
                stats = weather_df[col].describe()
                logging.info(f"{col}: mean={stats['mean']:.1f}, "
                           f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")
    
    def get_adapter_status(self) -> Dict:
        """Get status of weather adapters"""
        return {
            'primary_adapter': 'NOAA/NWS',
            'fallback_adapter': 'Open-Meteo',
            'usage_stats': self.adapter_usage.copy(),
            'available_parameters': {
                'noaa': self.primary_adapter.get_available_parameters(),
                'openmeteo': self.fallback_adapter.get_available_parameters()
            }
        }


class WeatherFeatureExtractor:
    """Weather feature extraction and processing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config['thresholds']
    
    def extract_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize weather features from raw weather data
        
        Args:
            weather_df: DataFrame with raw weather data
            
        Returns:
            DataFrame with extracted and normalized weather features
        """
        try:
            logging.info("Extracting weather features...")
            
            # Create copy to avoid modifying original
            features_df = weather_df.copy()
            
            # Extract thermal stress features
            features_df = self._extract_thermal_stress(features_df)
            
            # Extract wind stress features
            features_df = self._extract_wind_stress(features_df)
            
            # Extract precipitation stress features
            features_df = self._extract_precipitation_stress(features_df)
            
            # Extract storm features
            features_df = self._extract_storm_features(features_df)
            
            # Validate extracted features
            self._validate_features(features_df)
            
            # Create standardized output format
            output_df = self._create_standardized_output(features_df)
            
            logging.info(f"Weather feature extraction complete: {len(output_df)} records")
            return output_df
            
        except Exception as e:
            logging.error(f"Weather feature extraction failed: {e}")
            raise
    
    def _extract_thermal_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract thermal stress features (heat and cold stress)"""
        try:
            # Heat stress: 0 at HI≤85°F, 1 at HI≥100°F, linear interpolation
            heat_low = self.thresholds['thermal']['heat_low']
            heat_high = self.thresholds['thermal']['heat_high']
            
            df['heat_stress'] = np.clip(
                (df['heat_index'] - heat_low) / (heat_high - heat_low),
                0, 1
            )
            
            # Cold stress: 0 at T≥10°F, 1 at T≤0°F, linear interpolation
            cold_high = self.thresholds['thermal']['cold_low']  # 10°F
            cold_low = self.thresholds['thermal']['cold_high']   # 0°F
            
            df['cold_stress'] = np.clip(
                (cold_high - df['temp_2m']) / (cold_high - cold_low),
                0, 1
            )
            
            # Combined thermal stress (max of heat and cold)
            df['thermal_stress'] = np.maximum(df['heat_stress'], df['cold_stress'])
            
            return df
            
        except Exception as e:
            logging.error(f"Thermal stress extraction failed: {e}")
            raise
    
    def _extract_wind_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract wind stress features"""
        try:
            # Base wind gust scoring: 0 at ≤20mph, 1 at ≥50mph
            gust_low = self.thresholds['wind']['gust_low']
            gust_high = self.thresholds['wind']['gust_high']
            sustained_threshold = self.thresholds['wind']['sustained_threshold']
            
            # Base gust score
            df['gust_score'] = np.clip(
                (df['wind_gust'] - gust_low) / (gust_high - gust_low),
                0, 1
            )
            
            # Sustained wind bonus: +0.2 if sustained wind ≥30mph
            df['sustained_bonus'] = np.where(
                df['wind_speed'] >= sustained_threshold, 0.2, 0.0
            )
            
            # Combined wind stress (capped at 1.0)
            df['wind_stress'] = np.clip(
                df['gust_score'] + df['sustained_bonus'], 0, 1
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Wind stress extraction failed: {e}")
            raise
    
    def _extract_precipitation_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract precipitation stress features"""
        try:
            rain_heavy = self.thresholds['precip']['rain_heavy']
            snow_heavy = self.thresholds['precip']['snow_heavy']
            
            # Rain rate scoring: 0 at 0mm/h, 1 at ≥10mm/h
            df['rain_stress'] = np.clip(df['precip_rate'] / rain_heavy, 0, 1)
            
            # Snow rate scoring: 0 at 0cm/h, 1 at ≥5cm/h
            df['snow_stress'] = np.clip(df['snow_rate'] / snow_heavy, 0, 1)
            
            # Ice accretion: immediate maximum score (1.0) for any ice
            # Note: Ice detection would need additional data, using proxy for now
            df['ice_stress'] = 0.0  # Placeholder - would need ice accumulation data
            
            # Combined precipitation stress (max of rain, snow, ice)
            df['precipitation_stress'] = np.maximum.reduce([
                df['rain_stress'], df['snow_stress'], df['ice_stress']
            ])
            
            return df
            
        except Exception as e:
            logging.error(f"Precipitation stress extraction failed: {e}")
            raise
    
    def _extract_storm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract storm proxy features"""
        try:
            # Storm proxy: precipitation > 0 AND wind gust ≥35mph = 1.0
            storm_conditions = (df['precip_rate'] > 0) & (df['wind_gust'] >= 35)
            df['storm_proxy_binary'] = storm_conditions.astype(float)
            
            # Scaled storm scoring based on precipitation × wind gust product
            precip_norm = np.clip(df['precip_rate'] / 10.0, 0, 1)  # Normalize to 0-1
            wind_norm = np.clip((df['wind_gust'] - 20) / 30.0, 0, 1)  # 20-50 mph -> 0-1
            
            df['storm_proxy_scaled'] = precip_norm * wind_norm
            
            # Final storm proxy (max of binary and scaled)
            df['storm_proxy'] = np.maximum(df['storm_proxy_binary'], df['storm_proxy_scaled'])
            
            return df
            
        except Exception as e:
            logging.error(f"Storm feature extraction failed: {e}")
            raise
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate extracted weather features"""
        feature_cols = [
            'thermal_stress', 'wind_stress', 'precipitation_stress', 'storm_proxy'
        ]
        
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing feature column: {col}")
            
            # Check value range [0, 1]
            if df[col].min() < 0 or df[col].max() > 1:
                logging.warning(f"Feature {col} outside [0,1] range: "
                              f"[{df[col].min():.3f}, {df[col].max():.3f}]")
            
            # Check for excessive NaN values
            nan_pct = df[col].isna().mean() * 100
            if nan_pct > 50:
                logging.warning(f"High NaN percentage in {col}: {nan_pct:.1f}%")
    
    def _create_standardized_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized output format with required columns"""
        try:
            # Define output columns in standard order
            output_columns = [
                'cell_id', 'horizon_h', 'timestamp',
                'temp_2m', 'heat_index', 'wind_speed', 'wind_gust',
                'precip_rate', 'snow_rate', 'dewpoint', 'relative_humidity',
                'thermal_stress', 'wind_stress', 'precipitation_stress', 'storm_proxy',
                'confidence'
            ]
            
            # Select and reorder columns
            available_columns = [col for col in output_columns if col in df.columns]
            output_df = df[available_columns].copy()
            
            # Add missing columns with default values
            for col in output_columns:
                if col not in output_df.columns:
                    if col.endswith('_stress') or col == 'storm_proxy':
                        output_df[col] = 0.0
                    elif col == 'confidence':
                        output_df[col] = 0.5
                    else:
                        output_df[col] = np.nan
            
            # Reorder columns
            output_df = output_df[output_columns]
            
            # Sort by cell_id and horizon_h for consistency
            output_df = output_df.sort_values(['cell_id', 'horizon_h']).reset_index(drop=True)
            
            return output_df
            
        except Exception as e:
            logging.error(f"Standardized output creation failed: {e}")
            raise


# Weather Data Ingestion Tests
class WeatherIngestionTests:
    """Unit tests for weather data ingestion system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all weather ingestion tests"""
        logging.info("Running weather ingestion tests...")
        
        tests = [
            ('test_noaa_adapter_init', self.test_noaa_adapter_init),
            ('test_openmeteo_adapter_init', self.test_openmeteo_adapter_init),
            ('test_weather_manager_init', self.test_weather_manager_init),
            ('test_feature_extractor_init', self.test_feature_extractor_init),
            ('test_demo_weather_generation', self.test_demo_weather_generation),
            ('test_weather_validation', self.test_weather_validation),
            ('test_feature_extraction', self.test_feature_extraction),
            ('test_adapter_fallback_logic', self.test_adapter_fallback_logic)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✓ PASS" if result else "✗ FAIL"
                logging.info(f"{status}: {test_name}")
            except Exception as e:
                self.test_results[test_name] = False
                logging.error(f"✗ ERROR: {test_name} - {e}")
        
        # Summary
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        logging.info(f"Weather Ingestion Test Results: {passed}/{total} passed")
        
        return self.test_results
    
    def test_noaa_adapter_init(self) -> bool:
        """Test NOAA adapter initialization"""
        try:
            adapter = NOAAAdapter(self.config)
            
            # Check basic properties
            if not hasattr(adapter, 'base_url'):
                return False
            
            if not hasattr(adapter, 'session'):
                return False
            
            # Check available parameters
            params = adapter.get_available_parameters()
            if not isinstance(params, list) or len(params) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_openmeteo_adapter_init(self) -> bool:
        """Test Open-Meteo adapter initialization"""
        try:
            adapter = OpenMeteoAdapter(self.config)
            
            # Check basic properties
            if not hasattr(adapter, 'base_url'):
                return False
            
            if not hasattr(adapter, 'session'):
                return False
            
            # Check available parameters
            params = adapter.get_available_parameters()
            if not isinstance(params, list) or len(params) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_weather_manager_init(self) -> bool:
        """Test weather data manager initialization"""
        try:
            manager = WeatherDataManager(self.config)
            
            # Check adapters are initialized
            if not hasattr(manager, 'primary_adapter'):
                return False
            
            if not hasattr(manager, 'fallback_adapter'):
                return False
            
            # Check adapter types
            if not isinstance(manager.primary_adapter, NOAAAdapter):
                return False
            
            if not isinstance(manager.fallback_adapter, OpenMeteoAdapter):
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_feature_extractor_init(self) -> bool:
        """Test weather feature extractor initialization"""
        try:
            extractor = WeatherFeatureExtractor(self.config)
            
            # Check configuration is loaded
            if not hasattr(extractor, 'config'):
                return False
            
            if not hasattr(extractor, 'thresholds'):
                return False
            
            # Check required thresholds exist
            required_thresholds = ['thermal', 'wind', 'precip']
            for threshold in required_thresholds:
                if threshold not in extractor.thresholds:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def test_demo_weather_generation(self) -> bool:
        """Test demo weather data generation"""
        try:
            # Create minimal grid for testing
            test_grid = gpd.GeoDataFrame({
                'cell_id': ['test_001', 'test_002'],
                'centroid_lat': [40.0, 42.0],
                'centroid_lon': [-90.0, -88.0],
                'geometry': [Point(-90.0, 40.0), Point(-88.0, 42.0)]
            }, crs='EPSG:4326')
            
            manager = WeatherDataManager(self.config)
            demo_data = manager._generate_demo_weather_data(24, test_grid)
            
            # Check structure
            if not isinstance(demo_data, pd.DataFrame):
                return False
            
            if len(demo_data) == 0:
                return False
            
            # Check required columns
            required_cols = ['cell_id', 'horizon_h', 'temp_2m', 'wind_speed']
            for col in required_cols:
                if col not in demo_data.columns:
                    return False
            
            # Check data types and ranges
            if not demo_data['horizon_h'].eq(24).all():
                return False
            
            if demo_data['temp_2m'].isna().all():
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_weather_validation(self) -> bool:
        """Test weather data validation"""
        try:
            adapter = NOAAAdapter(self.config)
            
            # Test valid data
            valid_data = pd.DataFrame({
                'cell_id': ['test_001'],
                'horizon_h': [24],
                'temp_2m': [75.0],
                'wind_speed': [15.0],
                'precip_rate': [2.0]
            })
            
            is_valid, issues = adapter.validate_weather_data(valid_data)
            if not is_valid:
                return False
            
            # Test invalid data (missing columns)
            invalid_data = pd.DataFrame({
                'cell_id': ['test_001'],
                'horizon_h': [24]
                # Missing required columns
            })
            
            is_valid, issues = adapter.validate_weather_data(invalid_data)
            if is_valid:  # Should be invalid
                return False
            
            if len(issues) == 0:  # Should have issues
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test weather feature extraction"""
        try:
            # Create test weather data
            test_data = pd.DataFrame({
                'cell_id': ['test_001', 'test_002'],
                'horizon_h': [24, 24],
                'timestamp': [datetime.now(), datetime.now()],
                'temp_2m': [95.0, 30.0],  # Hot and cold
                'heat_index': [105.0, 30.0],
                'wind_speed': [25.0, 35.0],
                'wind_gust': [45.0, 55.0],  # High wind
                'precip_rate': [15.0, 0.0],  # Heavy rain
                'snow_rate': [0.0, 8.0],  # Heavy snow
                'dewpoint': [80.0, 20.0],
                'relative_humidity': [70.0, 60.0],
                'storm_probability': [0.8, 0.3],
                'confidence': [0.9, 0.8]
            })
            
            extractor = WeatherFeatureExtractor(self.config)
            features = extractor.extract_weather_features(test_data)
            
            # Check output structure
            if not isinstance(features, pd.DataFrame):
                return False
            
            if len(features) != 2:
                return False
            
            # Check feature columns exist
            feature_cols = ['thermal_stress', 'wind_stress', 'precipitation_stress', 'storm_proxy']
            for col in feature_cols:
                if col not in features.columns:
                    return False
            
            # Check feature ranges [0, 1]
            for col in feature_cols:
                if features[col].min() < 0 or features[col].max() > 1:
                    return False
            
            # Check specific feature logic
            # First row: hot temp should have high thermal stress
            if features.iloc[0]['thermal_stress'] < 0.5:
                return False
            
            # Second row: high wind should have high wind stress
            if features.iloc[1]['wind_stress'] < 0.5:
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_adapter_fallback_logic(self) -> bool:
        """Test adapter fallback logic"""
        try:
            manager = WeatherDataManager(self.config)
            
            # Check initial state
            if manager.adapter_usage['noaa'] != 0:
                return False
            
            if manager.adapter_usage['openmeteo'] != 0:
                return False
            
            # Check adapter status
            status = manager.get_adapter_status()
            
            if 'primary_adapter' not in status:
                return False
            
            if 'fallback_adapter' not in status:
                return False
            
            if status['primary_adapter'] != 'NOAA/NWS':
                return False
            
            if status['fallback_adapter'] != 'Open-Meteo':
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_test_summary(self) -> str:
        """Get formatted test summary"""
        if not self.test_results:
            return "No tests run yet"
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        summary = f"\n=== Weather Ingestion Test Summary ===\n"
        summary += f"Tests passed: {passed}/{total}\n\n"
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            summary += f"{status}: {test_name}\n"
        
        return summary