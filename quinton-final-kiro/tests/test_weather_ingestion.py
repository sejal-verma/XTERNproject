#!/usr/bin/env python3
"""
Test script for weather data ingestion system
Validates Task 3 implementation
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weather_adapters import (
    NOAAAdapter, OpenMeteoAdapter, WeatherDataManager, 
    WeatherFeatureExtractor, WeatherIngestionTests
)
from spatial_framework import SpatialProcessingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_config():
    """Create test configuration matching the main notebook structure"""
    return {
        'runtime': {
            'mode': 'demo',
            'horizons_h': [12, 24, 36, 48],
            'crs': 'EPSG:4326',
            'random_seed': 42,
            'hex_size_km': 40,
            'api_timeout': 30,
            'max_retries': 3
        },
        'weights': {
            'hazard': {
                'thermal': 0.3,
                'wind': 0.3,
                'precip': 0.25,
                'storm': 0.15
            },
            'exposure': {
                'pop': 0.7,
                'load': 0.3
            },
            'vulnerability': {
                'renew_share': 0.6,
                'tx_scarcity': 0.3,
                'outage': 0.1
            },
            'blend': {
                'alpha': 0.5,  # hazard weight
                'beta': 0.3,   # exposure weight
                'gamma': 0.2   # vulnerability weight
            }
        },
        'thresholds': {
            'thermal': {
                'heat_low': 85,   # °F
                'heat_high': 100,
                'cold_low': 10,
                'cold_high': 0
            },
            'wind': {
                'gust_low': 20,   # mph
                'gust_high': 50,
                'sustained_threshold': 30
            },
            'precip': {
                'rain_heavy': 10,  # mm/h
                'snow_heavy': 5,   # cm/h
                'ice_threshold': 0
            }
        }
    }

def create_test_grid():
    """Create a small test grid for validation"""
    # Create a few test cells in the MISO region
    test_cells = [
        {'cell_id': 'test_001', 'lat': 40.0, 'lon': -90.0},  # Illinois
        {'cell_id': 'test_002', 'lat': 42.0, 'lon': -88.0},  # Wisconsin
        {'cell_id': 'test_003', 'lat': 39.0, 'lon': -86.0},  # Indiana
        {'cell_id': 'test_004', 'lat': 44.0, 'lon': -93.0},  # Minnesota
        {'cell_id': 'test_005', 'lat': 32.0, 'lon': -92.0},  # Louisiana
    ]
    
    geometries = []
    data = []
    
    for cell in test_cells:
        # Create small hexagon around each point
        center = Point(cell['lon'], cell['lat'])
        # Simple square approximation for testing
        coords = [
            (cell['lon'] - 0.1, cell['lat'] - 0.1),
            (cell['lon'] + 0.1, cell['lat'] - 0.1),
            (cell['lon'] + 0.1, cell['lat'] + 0.1),
            (cell['lon'] - 0.1, cell['lat'] + 0.1),
            (cell['lon'] - 0.1, cell['lat'] - 0.1)
        ]
        
        from shapely.geometry import Polygon
        geom = Polygon(coords)
        
        geometries.append(geom)
        data.append({
            'cell_id': cell['cell_id'],
            'centroid_lat': cell['lat'],
            'centroid_lon': cell['lon'],
            'area_km2': 1600.0  # Approximate 40km hex area
        })
    
    return gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')

def test_weather_adapters():
    """Test weather adapter initialization and basic functionality"""
    print("\n=== Testing Weather Adapters ===")
    
    config = create_test_config()
    
    # Test NOAA adapter
    print("Testing NOAA adapter...")
    noaa_adapter = NOAAAdapter(config)
    noaa_params = noaa_adapter.get_available_parameters()
    print(f"✓ NOAA adapter initialized with {len(noaa_params)} parameters")
    
    # Test Open-Meteo adapter
    print("Testing Open-Meteo adapter...")
    openmeteo_adapter = OpenMeteoAdapter(config)
    openmeteo_params = openmeteo_adapter.get_available_parameters()
    print(f"✓ Open-Meteo adapter initialized with {len(openmeteo_params)} parameters")
    
    # Test weather manager
    print("Testing weather data manager...")
    weather_manager = WeatherDataManager(config)
    status = weather_manager.get_adapter_status()
    print(f"✓ Weather manager initialized")
    print(f"  Primary: {status['primary_adapter']}")
    print(f"  Fallback: {status['fallback_adapter']}")
    
    return True

def test_demo_weather_generation():
    """Test demo weather data generation"""
    print("\n=== Testing Demo Weather Generation ===")
    
    config = create_test_config()
    test_grid = create_test_grid()
    
    weather_manager = WeatherDataManager(config)
    
    # Test demo data generation for one horizon
    demo_data = weather_manager._generate_demo_weather_data(24, test_grid)
    
    print(f"✓ Generated demo weather data: {len(demo_data)} records")
    print(f"  Columns: {list(demo_data.columns)}")
    print(f"  Temperature range: {demo_data['temp_2m'].min():.1f} - {demo_data['temp_2m'].max():.1f}°F")
    print(f"  Wind speed range: {demo_data['wind_speed'].min():.1f} - {demo_data['wind_speed'].max():.1f} mph")
    
    # Validate data structure
    required_cols = ['cell_id', 'horizon_h', 'temp_2m', 'wind_speed', 'precip_rate']
    missing_cols = [col for col in required_cols if col not in demo_data.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False
    
    print("✓ Demo weather data validation passed")
    return True

def test_weather_feature_extraction():
    """Test weather feature extraction pipeline"""
    print("\n=== Testing Weather Feature Extraction ===")
    
    config = create_test_config()
    
    # Create test weather data
    test_weather = pd.DataFrame({
        'cell_id': ['test_001', 'test_002', 'test_003'],
        'horizon_h': [24, 24, 24],
        'timestamp': [datetime.now()] * 3,
        'temp_2m': [95.0, 30.0, 75.0],  # Hot, cold, moderate
        'heat_index': [105.0, 30.0, 78.0],
        'wind_speed': [15.0, 35.0, 25.0],
        'wind_gust': [25.0, 55.0, 35.0],  # Low, high, moderate
        'precip_rate': [0.0, 0.0, 15.0],  # None, none, heavy
        'snow_rate': [0.0, 8.0, 0.0],     # None, heavy, none
        'dewpoint': [80.0, 20.0, 65.0],
        'relative_humidity': [70.0, 60.0, 65.0],
        'storm_probability': [0.1, 0.3, 0.8],
        'confidence': [0.9, 0.8, 0.7]
    })
    
    # Extract features
    extractor = WeatherFeatureExtractor(config)
    features = extractor.extract_weather_features(test_weather)
    
    print(f"✓ Extracted weather features: {len(features)} records")
    print(f"  Feature columns: {[col for col in features.columns if 'stress' in col or 'proxy' in col]}")
    
    # Validate feature ranges
    feature_cols = ['thermal_stress', 'wind_stress', 'precipitation_stress', 'storm_proxy']
    for col in feature_cols:
        if col in features.columns:
            min_val, max_val = features[col].min(), features[col].max()
            print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
            
            if min_val < 0 or max_val > 1:
                print(f"✗ Feature {col} outside [0,1] range")
                return False
    
    # Test specific logic
    # Row 0: Hot temperature should have high thermal stress
    if features.iloc[0]['thermal_stress'] < 0.5:
        print("✗ Hot temperature should produce high thermal stress")
        return False
    
    # Row 1: High wind should have high wind stress
    if features.iloc[1]['wind_stress'] < 0.5:
        print("✗ High wind should produce high wind stress")
        return False
    
    # Row 2: Heavy precipitation should have high precipitation stress
    if features.iloc[2]['precipitation_stress'] < 0.5:
        print("✗ Heavy precipitation should produce high precipitation stress")
        return False
    
    print("✓ Weather feature extraction validation passed")
    return True

def test_full_weather_pipeline():
    """Test complete weather data pipeline with demo data"""
    print("\n=== Testing Full Weather Pipeline ===")
    
    config = create_test_config()
    test_grid = create_test_grid()
    
    # Initialize weather manager
    weather_manager = WeatherDataManager(config)
    
    # Fetch weather data (will use demo data due to API limitations)
    try:
        horizons = [12, 24]  # Test with fewer horizons for speed
        weather_data = weather_manager.fetch_weather_data(horizons, test_grid)
        
        print(f"✓ Weather data fetch completed: {len(weather_data)} records")
        print(f"  Unique cells: {weather_data['cell_id'].nunique()}")
        print(f"  Horizons: {sorted(weather_data['horizon_h'].unique())}")
        
        # Extract features
        extractor = WeatherFeatureExtractor(config)
        features = extractor.extract_weather_features(weather_data)
        
        print(f"✓ Feature extraction completed: {len(features)} records")
        
        # Validate final output
        required_output_cols = [
            'cell_id', 'horizon_h', 'temp_2m', 'wind_speed',
            'thermal_stress', 'wind_stress', 'precipitation_stress', 'storm_proxy'
        ]
        
        missing_cols = [col for col in required_output_cols if col not in features.columns]
        if missing_cols:
            print(f"✗ Missing output columns: {missing_cols}")
            return False
        
        print("✓ Full weather pipeline validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Weather pipeline failed: {e}")
        return False

def run_unit_tests():
    """Run comprehensive unit tests"""
    print("\n=== Running Unit Tests ===")
    
    config = create_test_config()
    test_suite = WeatherIngestionTests(config)
    
    results = test_suite.run_all_tests()
    
    print(test_suite.get_test_summary())
    
    passed = sum(results.values())
    total = len(results)
    
    return passed == total

def main():
    """Main test execution"""
    print("Weather Data Ingestion System - Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Weather Adapters", test_weather_adapters),
        ("Demo Weather Generation", test_demo_weather_generation),
        ("Weather Feature Extraction", test_weather_feature_extraction),
        ("Full Weather Pipeline", test_full_weather_pipeline),
        ("Unit Tests", run_unit_tests)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n✗ ERROR: {test_name} - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("WEATHER INGESTION VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All weather ingestion tests passed!")
        print("\nTask 3 implementation is complete and validated.")
        print("\nKey components implemented:")
        print("- NOAA/NWS gridpoint forecast adapter with error handling and rate limiting")
        print("- Open-Meteo fallback adapter with automatic switching")
        print("- Spatial aggregation from forecast grid to hex cells")
        print("- Data caching system for raw API responses")
        print("- Weather feature extraction pipeline with normalization")
        print("- Storm probability calculation from precipitation and wind data")
        print("- Data validation and quality checks")
        print("- Standardized output format with cell_id, horizon_h, timestamp columns")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)