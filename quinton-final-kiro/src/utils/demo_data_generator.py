"""
Demo Data Generator for MISO Weather-Stress Heatmap

This module generates realistic sample datasets for demo mode operation,
allowing the system to run without external API dependencies. The generated
data maintains realistic patterns and relationships for testing and demonstration.

Key Components:
- Weather forecast data generation
- Infrastructure and capacity data generation
- Population and demographic data generation
- Realistic spatial and temporal patterns
- Configurable data scenarios (normal, extreme weather, high stress)

Requirements addressed: 5.1, 7.4
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import json
import os


class DemoDataGenerator:
    """
    Generator for realistic demo data that maintains proper spatial and
    temporal relationships for testing the MISO weather-stress heatmap system.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize demo data generator.
        
        Args:
            random_seed: Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # MISO region approximate bounds (simplified)
        self.miso_bounds = {
            'min_lat': 37.0,
            'max_lat': 49.0,
            'min_lon': -104.0,
            'max_lon': -82.0
        }
        
        logging.info(f"Demo data generator initialized with seed {random_seed}")
    
    def generate_hex_grid_demo(self, n_cells: int = 500) -> gpd.GeoDataFrame:
        """
        Generate demo hexagonal grid for MISO footprint.
        
        Args:
            n_cells: Number of grid cells to generate
            
        Returns:
            GeoDataFrame with hexagonal grid cells
        """
        # Generate random points within MISO bounds
        lats = np.random.uniform(
            self.miso_bounds['min_lat'], 
            self.miso_bounds['max_lat'], 
            n_cells
        )
        lons = np.random.uniform(
            self.miso_bounds['min_lon'], 
            self.miso_bounds['max_lon'], 
            n_cells
        )
        
        # Create hexagonal cells (simplified as circles for demo)
        hex_size = 0.2  # Approximate hex radius in degrees
        
        cells = []
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            # Create approximate hexagon as circle
            center = Point(lon, lat)
            hex_cell = center.buffer(hex_size)
            
            cells.append({
                'cell_id': f'hex_{i:04d}',
                'geometry': hex_cell,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'area_km2': np.random.uniform(1200, 1800)  # ~40km hex area
            })
        
        grid_gdf = gpd.GeoDataFrame(cells, crs='EPSG:4326')
        
        logging.info(f"Generated demo hex grid with {len(grid_gdf)} cells")
        return grid_gdf
    
    def generate_weather_demo_data(self, 
                                  grid: gpd.GeoDataFrame,
                                  horizons_h: List[int] = [12, 24, 36, 48],
                                  scenario: str = 'normal') -> pd.DataFrame:
        """
        Generate demo weather forecast data for grid cells.
        
        Args:
            grid: Hexagonal grid GeoDataFrame
            horizons_h: Forecast horizons in hours
            scenario: Weather scenario ('normal', 'heat_wave', 'winter_storm', 'severe_weather')
            
        Returns:
            DataFrame with weather forecast data
        """
        weather_data = []
        base_time = datetime.now()
        
        for _, cell in grid.iterrows():
            cell_id = cell['cell_id']
            lat = cell['centroid_lat']
            lon = cell['centroid_lon']
            
            # Generate base weather patterns based on location
            base_temp_f = self._get_base_temperature(lat, scenario)
            base_humidity = self._get_base_humidity(lat, lon)
            
            for horizon_h in horizons_h:
                forecast_time = base_time + timedelta(hours=horizon_h)
                
                # Generate weather parameters with realistic variations
                weather_params = self._generate_weather_parameters(
                    base_temp_f, base_humidity, lat, lon, horizon_h, scenario
                )
                
                weather_record = {
                    'cell_id': cell_id,
                    'horizon_h': horizon_h,
                    'timestamp': forecast_time,
                    'centroid_lat': lat,
                    'centroid_lon': lon,
                    **weather_params
                }
                
                weather_data.append(weather_record)
        
        weather_df = pd.DataFrame(weather_data)
        
        logging.info(f"Generated demo weather data: {len(weather_df)} records, scenario: {scenario}")
        return weather_df
    
    def _get_base_temperature(self, lat: float, scenario: str) -> float:
        """Get base temperature based on latitude and scenario"""
        # Temperature decreases with latitude (roughly)
        base_temp_c = 25 - (lat - 37) * 1.5  # Celsius
        
        # Scenario adjustments
        if scenario == 'heat_wave':
            base_temp_c += np.random.uniform(8, 15)
        elif scenario == 'winter_storm':
            base_temp_c -= np.random.uniform(15, 25)
        elif scenario == 'severe_weather':
            base_temp_c += np.random.uniform(-5, 10)
        
        # Convert to Fahrenheit
        return base_temp_c * 9/5 + 32
    
    def _get_base_humidity(self, lat: float, lon: float) -> float:
        """Get base humidity based on location"""
        # Higher humidity in eastern regions and southern areas
        humidity = 0.4 + (lon + 104) / 44 * 0.3  # East-west gradient
        humidity += (49 - lat) / 12 * 0.2  # North-south gradient
        return np.clip(humidity, 0.2, 0.9)
    
    def _generate_weather_parameters(self, 
                                   base_temp_f: float,
                                   base_humidity: float,
                                   lat: float, lon: float,
                                   horizon_h: int,
                                   scenario: str) -> Dict[str, float]:
        """Generate complete weather parameter set"""
        
        # Temperature with diurnal variation and uncertainty
        temp_variation = np.sin((horizon_h % 24) * np.pi / 12) * 8  # Diurnal cycle
        temp_uncertainty = np.random.normal(0, 2 + horizon_h * 0.1)  # Increasing uncertainty
        temp_2m = base_temp_f + temp_variation + temp_uncertainty
        
        # Heat index calculation (simplified)
        heat_index = self._calculate_heat_index(temp_2m, base_humidity)
        
        # Wind parameters
        wind_speed, wind_gust = self._generate_wind_parameters(scenario, lat, lon)
        
        # Precipitation parameters
        precip_rate, snow_rate, ice_rate = self._generate_precipitation_parameters(
            temp_2m, scenario, lat, lon
        )
        
        # Dewpoint and relative humidity
        dewpoint = temp_2m - np.random.uniform(5, 25)
        relative_humidity = base_humidity + np.random.normal(0, 0.1)
        relative_humidity = np.clip(relative_humidity, 0.1, 0.95)
        
        # Storm probability
        storm_probability = self._calculate_storm_probability(
            precip_rate, wind_gust, scenario
        )
        
        return {
            'temp_2m': temp_2m,
            'heat_index': heat_index,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust,
            'precip_rate': precip_rate,
            'snow_rate': snow_rate,
            'ice_rate': ice_rate,
            'dewpoint': dewpoint,
            'relative_humidity': relative_humidity,
            'storm_probability': storm_probability
        }
    
    def _calculate_heat_index(self, temp_f: float, humidity: float) -> float:
        """Calculate heat index (simplified formula)"""
        if temp_f < 80:
            return temp_f
        
        # Simplified heat index calculation
        hi = (temp_f + humidity * 100) * 0.6 + temp_f * 0.4
        return max(temp_f, hi)
    
    def _generate_wind_parameters(self, scenario: str, lat: float, lon: float) -> Tuple[float, float]:
        """Generate wind speed and gust parameters"""
        # Base wind patterns
        base_wind = np.random.uniform(5, 15)  # mph
        
        # Scenario adjustments
        if scenario == 'severe_weather':
            base_wind += np.random.uniform(15, 35)
        elif scenario == 'winter_storm':
            base_wind += np.random.uniform(10, 25)
        
        # Geographic variations (higher winds in plains)
        if -100 < lon < -90:  # Great Plains region
            base_wind *= np.random.uniform(1.2, 1.8)
        
        wind_speed = max(0, base_wind + np.random.normal(0, 3))
        wind_gust = wind_speed * np.random.uniform(1.2, 2.0)
        
        return wind_speed, wind_gust
    
    def _generate_precipitation_parameters(self, 
                                         temp_f: float,
                                         scenario: str,
                                         lat: float, lon: float) -> Tuple[float, float, float]:
        """Generate precipitation parameters"""
        
        # Base precipitation probability
        precip_prob = 0.1
        
        # Scenario adjustments
        if scenario in ['winter_storm', 'severe_weather']:
            precip_prob = 0.7
        elif scenario == 'heat_wave':
            precip_prob = 0.05
        
        # Generate precipitation
        if np.random.random() < precip_prob:
            if temp_f > 35:  # Rain
                precip_rate = np.random.exponential(2.0)  # mm/h
                snow_rate = 0.0
            else:  # Snow
                precip_rate = 0.0
                snow_rate = np.random.exponential(1.0)  # cm/h
            
            # Ice (rare)
            ice_rate = 0.1 if (temp_f < 35 and np.random.random() < 0.05) else 0.0
        else:
            precip_rate = snow_rate = ice_rate = 0.0
        
        return precip_rate, snow_rate, ice_rate
    
    def _calculate_storm_probability(self, 
                                   precip_rate: float,
                                   wind_gust: float,
                                   scenario: str) -> float:
        """Calculate storm probability"""
        storm_prob = 0.0
        
        if precip_rate > 0 and wind_gust > 25:
            storm_prob = min(1.0, (precip_rate / 5.0) * (wind_gust / 40.0))
        
        if scenario == 'severe_weather':
            storm_prob = max(storm_prob, 0.6)
        
        return storm_prob
    
    def generate_infrastructure_demo_data(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Generate demo infrastructure and capacity data.
        
        Args:
            grid: Hexagonal grid GeoDataFrame
            
        Returns:
            DataFrame with infrastructure data
        """
        infrastructure_data = []
        
        for _, cell in grid.iterrows():
            cell_id = cell['cell_id']
            lat = cell['centroid_lat']
            lon = cell['centroid_lon']
            
            # Population density (higher near urban centers)
            pop_density = self._generate_population_density(lat, lon)
            
            # Generation capacity data
            capacity_data = self._generate_capacity_data(lat, lon)
            
            # Transmission data
            transmission_data = self._generate_transmission_data(lat, lon)
            
            infrastructure_record = {
                'cell_id': cell_id,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'population_density': pop_density,
                'normalized_pop_density': self._normalize_population(pop_density),
                **capacity_data,
                **transmission_data
            }
            
            infrastructure_data.append(infrastructure_record)
        
        infrastructure_df = pd.DataFrame(infrastructure_data)
        
        logging.info(f"Generated demo infrastructure data: {len(infrastructure_df)} records")
        return infrastructure_df
    
    def _generate_population_density(self, lat: float, lon: float) -> float:
        """Generate population density with urban clustering"""
        # Define some "urban centers" for realistic clustering
        urban_centers = [
            (41.8781, -87.6298),  # Chicago area
            (39.7392, -104.9903), # Denver area
            (44.9778, -93.2650),  # Minneapolis area
            (39.0458, -76.6413),  # Baltimore area
        ]
        
        # Calculate distance to nearest urban center
        min_distance = float('inf')
        for urban_lat, urban_lon in urban_centers:
            distance = np.sqrt((lat - urban_lat)**2 + (lon - urban_lon)**2)
            min_distance = min(min_distance, distance)
        
        # Population density decreases with distance from urban centers
        base_density = 50  # people per km²
        urban_factor = np.exp(-min_distance * 2)  # Exponential decay
        
        pop_density = base_density * (1 + urban_factor * 20) * np.random.uniform(0.5, 2.0)
        return max(1, pop_density)
    
    def _normalize_population(self, pop_density: float) -> float:
        """Normalize population density to [0, 1] scale"""
        # Assume max density of 1000 people/km² for normalization
        return min(1.0, pop_density / 1000.0)
    
    def _generate_capacity_data(self, lat: float, lon: float) -> Dict[str, float]:
        """Generate generation capacity data"""
        
        # Wind capacity (higher in plains states)
        wind_factor = 1.0
        if -100 < lon < -90:  # Great Plains
            wind_factor = 3.0
        wind_capacity = np.random.exponential(100) * wind_factor
        
        # Solar capacity (higher in southern regions)
        solar_factor = max(0.5, (45 - lat) / 8)  # More solar in south
        solar_capacity = np.random.exponential(50) * solar_factor
        
        # Thermal capacity (coal, gas, nuclear)
        thermal_capacity = np.random.exponential(200)
        
        # Calculate renewable share
        total_capacity = wind_capacity + solar_capacity + thermal_capacity
        renewable_share = (wind_capacity + solar_capacity) / total_capacity if total_capacity > 0 else 0
        
        return {
            'wind_capacity_mw': wind_capacity,
            'solar_capacity_mw': solar_capacity,
            'thermal_capacity_mw': thermal_capacity,
            'total_capacity_mw': total_capacity,
            'renewable_share': renewable_share
        }
    
    def _generate_transmission_data(self, lat: float, lon: float) -> Dict[str, float]:
        """Generate transmission infrastructure data"""
        
        # Transmission density (higher near load centers and between regions)
        base_density = np.random.uniform(0.2, 0.8)
        
        # Higher density in corridor regions
        if 39 < lat < 42 and -90 < lon < -85:  # Midwest corridor
            base_density *= 1.5
        
        transmission_scarcity = 1.0 - base_density  # Invert for scarcity metric
        
        return {
            'transmission_density': base_density,
            'transmission_scarcity': transmission_scarcity,
            'outage_flag': np.random.random() < 0.05  # 5% chance of outage
        }
    
    def save_demo_datasets(self, 
                          output_dir: str = 'data/demo',
                          scenarios: List[str] = ['normal', 'heat_wave', 'winter_storm']) -> Dict[str, str]:
        """
        Generate and save complete demo datasets for multiple scenarios.
        
        Args:
            output_dir: Directory to save demo data
            scenarios: List of weather scenarios to generate
            
        Returns:
            Dictionary mapping scenario names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base grid (same for all scenarios)
        grid = self.generate_hex_grid_demo(n_cells=200)  # Smaller for demo
        grid_path = os.path.join(output_dir, 'demo_hex_grid.geojson')
        grid.to_file(grid_path, driver='GeoJSON')
        
        # Generate infrastructure data (same for all scenarios)
        infrastructure = self.generate_infrastructure_demo_data(grid)
        infra_path = os.path.join(output_dir, 'demo_infrastructure.csv')
        infrastructure.to_csv(infra_path, index=False)
        
        # Generate weather data for each scenario
        saved_files = {
            'grid': grid_path,
            'infrastructure': infra_path,
            'weather_scenarios': {}
        }
        
        for scenario in scenarios:
            weather_data = self.generate_weather_demo_data(grid, scenario=scenario)
            weather_path = os.path.join(output_dir, f'demo_weather_{scenario}.csv')
            weather_data.to_csv(weather_path, index=False)
            saved_files['weather_scenarios'][scenario] = weather_path
        
        # Generate metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'n_cells': len(grid),
            'scenarios': scenarios,
            'description': 'Demo datasets for MISO Weather-Stress Heatmap system'
        }
        
        metadata_path = os.path.join(output_dir, 'demo_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = metadata_path
        
        logging.info(f"Saved demo datasets to {output_dir}")
        logging.info(f"Generated {len(scenarios)} weather scenarios for {len(grid)} grid cells")
        
        return saved_files
    
    def create_performance_benchmark_data(self, 
                                        n_cells_list: List[int] = [100, 500, 1000, 2000]) -> Dict[str, pd.DataFrame]:
        """
        Create datasets of varying sizes for performance benchmarking.
        
        Args:
            n_cells_list: List of grid sizes to generate
            
        Returns:
            Dictionary mapping grid sizes to datasets
        """
        benchmark_data = {}
        
        for n_cells in n_cells_list:
            logging.info(f"Generating benchmark data for {n_cells} cells")
            
            # Generate grid and data
            grid = self.generate_hex_grid_demo(n_cells=n_cells)
            weather = self.generate_weather_demo_data(grid, scenario='normal')
            infrastructure = self.generate_infrastructure_demo_data(grid)
            
            benchmark_data[f'{n_cells}_cells'] = {
                'grid': grid,
                'weather': weather,
                'infrastructure': infrastructure,
                'n_cells': n_cells,
                'n_weather_records': len(weather),
                'n_infra_records': len(infrastructure)
            }
        
        logging.info(f"Generated benchmark datasets for {len(n_cells_list)} different grid sizes")
        return benchmark_data


def create_demo_configuration() -> Dict[str, Any]:
    """
    Create configuration for demo mode operation.
    
    Returns:
        Demo configuration dictionary
    """
    return {
        'runtime': {
            'mode': 'demo',
            'horizons_h': [12, 24, 36, 48],
            'crs': 'EPSG:4326',
            'random_seed': 42
        },
        'demo_settings': {
            'scenario': 'normal',  # 'normal', 'heat_wave', 'winter_storm', 'severe_weather'
            'n_cells': 200,
            'data_dir': 'data/demo',
            'use_cached_data': True
        },
        'weights': {
            'thermal': 0.3,
            'wind': 0.3,
            'precip': 0.25,
            'storm': 0.15,
            'pop': 0.7,
            'load': 0.3,
            'renew_share': 0.6,
            'tx_scarcity': 0.3,
            'outage': 0.1,
            'alpha': 0.5,
            'beta': 0.3,
            'gamma': 0.2
        }
    }


def validate_demo_data_quality(weather_df: pd.DataFrame, 
                             infrastructure_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the quality and realism of generated demo data.
    
    Args:
        weather_df: Generated weather data
        infrastructure_df: Generated infrastructure data
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {
        'weather_validation': {},
        'infrastructure_validation': {},
        'overall_quality': True
    }
    
    # Weather data validation
    weather_val = validation_results['weather_validation']
    
    # Check temperature ranges
    temp_range = (weather_df['temp_2m'].min(), weather_df['temp_2m'].max())
    weather_val['temperature_range_realistic'] = -20 <= temp_range[0] <= 120 and temp_range[1] <= 120
    
    # Check heat index relationship
    weather_val['heat_index_valid'] = (weather_df['heat_index'] >= weather_df['temp_2m']).all()
    
    # Check wind relationships
    weather_val['wind_gust_higher'] = (weather_df['wind_gust'] >= weather_df['wind_speed']).all()
    
    # Check precipitation non-negative
    weather_val['precipitation_non_negative'] = (
        (weather_df['precip_rate'] >= 0).all() and
        (weather_df['snow_rate'] >= 0).all() and
        (weather_df['ice_rate'] >= 0).all()
    )
    
    # Infrastructure data validation
    infra_val = validation_results['infrastructure_validation']
    
    # Check renewable share range
    infra_val['renewable_share_valid'] = (
        (infrastructure_df['renewable_share'] >= 0).all() and
        (infrastructure_df['renewable_share'] <= 1).all()
    )
    
    # Check population density
    infra_val['population_positive'] = (infrastructure_df['population_density'] > 0).all()
    
    # Check capacity consistency
    total_calc = (infrastructure_df['wind_capacity_mw'] + 
                 infrastructure_df['solar_capacity_mw'] + 
                 infrastructure_df['thermal_capacity_mw'])
    infra_val['capacity_consistent'] = np.allclose(
        total_calc, infrastructure_df['total_capacity_mw'], rtol=1e-3
    )
    
    # Overall quality assessment
    all_checks = list(weather_val.values()) + list(infra_val.values())
    validation_results['overall_quality'] = all(all_checks)
    
    return validation_results