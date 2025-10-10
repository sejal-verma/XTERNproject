"""
Unit tests for feature engineering and normalization system.

Tests validate threshold behavior, edge cases, and mathematical correctness
of all stress calculation functions according to the requirements.
"""

import unittest
import numpy as np
import pandas as pd
from feature_engineering import (
    FeatureEngineeringEngine,
    ThermalThresholds,
    WindThresholds,
    PrecipThresholds,
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    mph_to_ms,
    ms_to_mph
)


class TestThermalStressCalculation(unittest.TestCase):
    """Test thermal stress calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FeatureEngineeringEngine()
        self.thermal_thresholds = self.engine.thermal_thresholds
    
    def test_heat_stress_thresholds(self):
        """Test heat stress scoring at exact thresholds"""
        # Test at lower threshold (85°F) - should be 0
        heat_stress_low = self.engine.calculate_thermal_stress(90.0, 85.0)
        self.assertAlmostEqual(heat_stress_low, 0.0, places=3)
        
        # Test at upper threshold (100°F) - should be 1
        heat_stress_high = self.engine.calculate_thermal_stress(90.0, 100.0)
        self.assertAlmostEqual(heat_stress_high, 1.0, places=3)
        
        # Test midpoint (92.5°F) - should be 0.5
        heat_stress_mid = self.engine.calculate_thermal_stress(90.0, 92.5)
        self.assertAlmostEqual(heat_stress_mid, 0.5, places=3)
    
    def test_cold_stress_thresholds(self):
        """Test cold stress scoring at exact thresholds"""
        # Test at upper threshold (10°F) - should be 0
        cold_stress_high = self.engine.calculate_thermal_stress(10.0, 70.0)
        self.assertAlmostEqual(cold_stress_high, 0.0, places=3)
        
        # Test at lower threshold (0°F) - should be 1
        cold_stress_low = self.engine.calculate_thermal_stress(0.0, 70.0)
        self.assertAlmostEqual(cold_stress_low, 1.0, places=3)
        
        # Test midpoint (5°F) - should be 0.5
        cold_stress_mid = self.engine.calculate_thermal_stress(5.0, 70.0)
        self.assertAlmostEqual(cold_stress_mid, 0.5, places=3)
    
    def test_thermal_stress_maximum_selection(self):
        """Test that thermal stress returns maximum of heat and cold stress"""
        # High heat, low cold - should return heat stress
        high_heat_low_cold = self.engine.calculate_thermal_stress(50.0, 95.0)
        expected_heat = (95.0 - 85.0) / (100.0 - 85.0)  # 0.667
        self.assertAlmostEqual(high_heat_low_cold, expected_heat, places=3)
        
        # Low heat, high cold - should return cold stress
        low_heat_high_cold = self.engine.calculate_thermal_stress(-5.0, 70.0)
        expected_cold = (10.0 - (-5.0)) / (10.0 - 0.0)  # 1.0 (capped)
        self.assertAlmostEqual(low_heat_high_cold, 1.0, places=3)
        
        # Both moderate - should return higher of the two
        moderate_both = self.engine.calculate_thermal_stress(7.0, 88.0)
        expected_heat = (88.0 - 85.0) / (100.0 - 85.0)  # 0.2
        expected_cold = (10.0 - 7.0) / (10.0 - 0.0)     # 0.3
        self.assertAlmostEqual(moderate_both, max(expected_heat, expected_cold), places=3)
    
    def test_thermal_stress_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Extreme heat beyond threshold
        extreme_heat = self.engine.calculate_thermal_stress(50.0, 120.0)
        self.assertAlmostEqual(extreme_heat, 1.0, places=3)
        
        # Extreme cold beyond threshold
        extreme_cold = self.engine.calculate_thermal_stress(-20.0, 70.0)
        self.assertAlmostEqual(extreme_cold, 1.0, places=3)
        
        # Normal conditions (no stress)
        normal_conditions = self.engine.calculate_thermal_stress(70.0, 80.0)
        self.assertAlmostEqual(normal_conditions, 0.0, places=3)
    
    def test_thermal_stress_array_input(self):
        """Test thermal stress calculation with numpy arrays"""
        temperatures = np.array([0.0, 5.0, 10.0, 70.0])
        heat_indices = np.array([70.0, 70.0, 70.0, 100.0])
        
        results = self.engine.calculate_thermal_stress(temperatures, heat_indices)
        
        # Expected: [1.0, 0.5, 0.0, 1.0] (cold, cold, none, heat)
        expected = np.array([1.0, 0.5, 0.0, 1.0])
        np.testing.assert_array_almost_equal(results, expected, decimal=3)
    
    def test_thermal_stress_pandas_series(self):
        """Test thermal stress calculation with pandas Series"""
        df = pd.DataFrame({
            'temp': [0.0, 5.0, 10.0, 70.0],
            'heat_index': [70.0, 70.0, 70.0, 100.0]
        })
        
        results = self.engine.calculate_thermal_stress(df['temp'], df['heat_index'])
        
        # Expected: [1.0, 0.5, 0.0, 1.0]
        expected = pd.Series([1.0, 0.5, 0.0, 1.0])
        pd.testing.assert_series_equal(results, expected, check_names=False, atol=1e-3)


class TestWindStressCalculation(unittest.TestCase):
    """Test wind stress calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FeatureEngineeringEngine()
        self.wind_thresholds = self.engine.wind_thresholds
    
    def test_wind_gust_thresholds(self):
        """Test wind gust scoring at exact thresholds"""
        # Test at lower threshold (20 mph) - should be 0
        wind_stress_low = self.engine.calculate_wind_stress(10.0, 20.0)
        self.assertAlmostEqual(wind_stress_low, 0.0, places=3)
        
        # Test at upper threshold (50 mph) - should be 1
        wind_stress_high = self.engine.calculate_wind_stress(10.0, 50.0)
        self.assertAlmostEqual(wind_stress_high, 1.0, places=3)
        
        # Test midpoint (35 mph) - should be 0.5
        wind_stress_mid = self.engine.calculate_wind_stress(10.0, 35.0)
        self.assertAlmostEqual(wind_stress_mid, 0.5, places=3)
    
    def test_sustained_wind_bonus(self):
        """Test sustained wind bonus application"""
        # Base gust stress + sustained wind bonus
        base_gust = 30.0  # 30 mph gust = (30-20)/(50-20) = 0.333
        sustained_wind = 35.0  # Above 30 mph threshold = +0.2 bonus
        
        wind_stress = self.engine.calculate_wind_stress(sustained_wind, base_gust)
        expected = (30.0 - 20.0) / (50.0 - 20.0) + 0.2  # 0.333 + 0.2 = 0.533
        self.assertAlmostEqual(wind_stress, expected, places=3)
        
        # No sustained wind bonus
        wind_stress_no_bonus = self.engine.calculate_wind_stress(25.0, base_gust)
        expected_no_bonus = (30.0 - 20.0) / (50.0 - 20.0)  # 0.333
        self.assertAlmostEqual(wind_stress_no_bonus, expected_no_bonus, places=3)
    
    def test_wind_stress_capping(self):
        """Test that wind stress is capped at 1.0"""
        # High gust + sustained wind bonus should cap at 1.0
        high_gust = 60.0  # Above threshold
        high_sustained = 40.0  # Above threshold
        
        wind_stress = self.engine.calculate_wind_stress(high_sustained, high_gust)
        self.assertAlmostEqual(wind_stress, 1.0, places=3)
    
    def test_wind_stress_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # No wind
        no_wind = self.engine.calculate_wind_stress(0.0, 0.0)
        self.assertAlmostEqual(no_wind, 0.0, places=3)
        
        # Sustained wind exactly at threshold
        at_threshold = self.engine.calculate_wind_stress(30.0, 25.0)
        expected = (25.0 - 20.0) / (50.0 - 20.0) + 0.2  # 0.167 + 0.2 = 0.367
        self.assertAlmostEqual(at_threshold, expected, places=3)
    
    def test_wind_stress_array_input(self):
        """Test wind stress calculation with numpy arrays"""
        wind_speeds = np.array([10.0, 25.0, 35.0, 40.0])
        wind_gusts = np.array([15.0, 30.0, 40.0, 60.0])
        
        results = self.engine.calculate_wind_stress(wind_speeds, wind_gusts)
        
        # Calculate expected values
        base_stresses = np.clip((wind_gusts - 20.0) / (50.0 - 20.0), 0.0, 1.0)
        bonuses = np.where(wind_speeds >= 30.0, 0.2, 0.0)
        expected = np.minimum(base_stresses + bonuses, 1.0)
        
        np.testing.assert_array_almost_equal(results, expected, decimal=3)


class TestPrecipitationStressCalculation(unittest.TestCase):
    """Test precipitation stress calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FeatureEngineeringEngine()
        self.precip_thresholds = self.engine.precip_thresholds
    
    def test_rain_stress_thresholds(self):
        """Test rain stress scoring at exact thresholds"""
        # Test at zero rain - should be 0
        rain_stress_zero = self.engine.calculate_precipitation_stress(0.0, 0.0, 0.0)
        self.assertAlmostEqual(rain_stress_zero, 0.0, places=3)
        
        # Test at heavy rain threshold (10 mm/h) - should be 1
        rain_stress_heavy = self.engine.calculate_precipitation_stress(10.0, 0.0, 0.0)
        self.assertAlmostEqual(rain_stress_heavy, 1.0, places=3)
        
        # Test at midpoint (5 mm/h) - should be 0.5
        rain_stress_mid = self.engine.calculate_precipitation_stress(5.0, 0.0, 0.0)
        self.assertAlmostEqual(rain_stress_mid, 0.5, places=3)
    
    def test_snow_stress_thresholds(self):
        """Test snow stress scoring at exact thresholds"""
        # Test at zero snow - should be 0
        snow_stress_zero = self.engine.calculate_precipitation_stress(0.0, 0.0, 0.0)
        self.assertAlmostEqual(snow_stress_zero, 0.0, places=3)
        
        # Test at heavy snow threshold (5 cm/h) - should be 1
        snow_stress_heavy = self.engine.calculate_precipitation_stress(0.0, 5.0, 0.0)
        self.assertAlmostEqual(snow_stress_heavy, 1.0, places=3)
        
        # Test at midpoint (2.5 cm/h) - should be 0.5
        snow_stress_mid = self.engine.calculate_precipitation_stress(0.0, 2.5, 0.0)
        self.assertAlmostEqual(snow_stress_mid, 0.5, places=3)
    
    def test_ice_stress_immediate_maximum(self):
        """Test that any ice accumulation gives maximum score"""
        # Any ice should give 1.0
        ice_stress_minimal = self.engine.calculate_precipitation_stress(0.0, 0.0, 0.1)
        self.assertAlmostEqual(ice_stress_minimal, 1.0, places=3)
        
        ice_stress_heavy = self.engine.calculate_precipitation_stress(0.0, 0.0, 5.0)
        self.assertAlmostEqual(ice_stress_heavy, 1.0, places=3)
    
    def test_precipitation_stress_maximum_selection(self):
        """Test that precipitation stress returns maximum of all types"""
        # Rain dominant
        rain_dominant = self.engine.calculate_precipitation_stress(8.0, 1.0, 0.0)
        expected_rain = 8.0 / 10.0  # 0.8
        expected_snow = 1.0 / 5.0   # 0.2
        self.assertAlmostEqual(rain_dominant, max(expected_rain, expected_snow, 0.0), places=3)
        
        # Snow dominant
        snow_dominant = self.engine.calculate_precipitation_stress(2.0, 4.0, 0.0)
        expected_rain = 2.0 / 10.0  # 0.2
        expected_snow = 4.0 / 5.0   # 0.8
        self.assertAlmostEqual(snow_dominant, max(expected_rain, expected_snow, 0.0), places=3)
        
        # Ice dominant (always 1.0)
        ice_dominant = self.engine.calculate_precipitation_stress(2.0, 1.0, 0.1)
        self.assertAlmostEqual(ice_dominant, 1.0, places=3)
    
    def test_precipitation_stress_array_input(self):
        """Test precipitation stress calculation with numpy arrays"""
        rain_rates = np.array([0.0, 5.0, 10.0, 15.0])
        snow_rates = np.array([0.0, 2.5, 0.0, 0.0])
        ice_rates = np.array([0.0, 0.0, 0.0, 0.1])
        
        results = self.engine.calculate_precipitation_stress(rain_rates, snow_rates, ice_rates)
        
        # Expected: [0.0, 0.5, 1.0, 1.0] (none, snow, rain, ice)
        expected = np.array([0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(results, expected, decimal=3)


class TestStormProxyCalculation(unittest.TestCase):
    """Test storm proxy calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FeatureEngineeringEngine()
    
    def test_full_storm_conditions(self):
        """Test full storm detection (precip > 0 AND gust >= 35mph)"""
        # Full storm conditions
        full_storm = self.engine.calculate_storm_proxy(5.0, 40.0)
        self.assertAlmostEqual(full_storm, 1.0, places=3)
        
        # Precipitation but insufficient wind
        no_storm_wind = self.engine.calculate_storm_proxy(5.0, 30.0)
        self.assertLess(no_storm_wind, 1.0)
        
        # Wind but no precipitation
        no_storm_precip = self.engine.calculate_storm_proxy(0.0, 40.0)
        self.assertAlmostEqual(no_storm_precip, 0.0, places=3)
    
    def test_scaled_storm_scoring(self):
        """Test scaled scoring for partial storm conditions"""
        # Moderate precipitation and wind
        precip_rate = 2.5  # Normalized: 2.5/5.0 = 0.5
        wind_gust = 25.0   # Normalized: 25.0/50.0 = 0.5
        
        storm_score = self.engine.calculate_storm_proxy(precip_rate, wind_gust)
        expected = 0.5 * 0.5  # 0.25
        self.assertAlmostEqual(storm_score, expected, places=3)
    
    def test_storm_proxy_edge_cases(self):
        """Test edge cases for storm proxy calculation"""
        # No conditions
        no_conditions = self.engine.calculate_storm_proxy(0.0, 0.0)
        self.assertAlmostEqual(no_conditions, 0.0, places=3)
        
        # Extreme conditions (should cap at 1.0 for full storm)
        extreme_conditions = self.engine.calculate_storm_proxy(20.0, 60.0)
        self.assertAlmostEqual(extreme_conditions, 1.0, places=3)
    
    def test_storm_proxy_array_input(self):
        """Test storm proxy calculation with numpy arrays"""
        precip_rates = np.array([0.0, 2.0, 5.0, 10.0])
        wind_gusts = np.array([20.0, 30.0, 40.0, 45.0])
        
        results = self.engine.calculate_storm_proxy(precip_rates, wind_gusts)
        
        # Calculate expected values
        # [0.0, scaled, full_storm, full_storm]
        expected = np.array([0.0, 0.4 * 0.6, 1.0, 1.0])  # Approximate
        
        # Check that results are reasonable
        self.assertAlmostEqual(results[0], 0.0, places=3)  # No conditions
        self.assertLess(results[1], 1.0)  # Partial conditions
        self.assertAlmostEqual(results[2], 1.0, places=3)  # Full storm
        self.assertAlmostEqual(results[3], 1.0, places=3)  # Full storm


class TestFeatureEngineeringIntegration(unittest.TestCase):
    """Test complete feature engineering pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FeatureEngineeringEngine()
        
        # Create sample weather data
        self.sample_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C', 'D'],
            'temp_2m': [0.0, 70.0, 90.0, 5.0],  # Cold, normal, hot, cold
            'heat_index': [70.0, 75.0, 105.0, 70.0],  # Normal, normal, extreme, normal
            'wind_speed': [10.0, 25.0, 35.0, 40.0],  # Low, moderate, high, high
            'wind_gust': [15.0, 30.0, 45.0, 55.0],  # Low, moderate, high, extreme
            'precip_rate': [0.0, 2.0, 8.0, 0.0],  # None, light, heavy, none
            'snow_rate': [0.0, 0.0, 0.0, 3.0],  # None, none, none, moderate
            'ice_rate': [0.0, 0.0, 0.0, 0.1]  # None, none, none, any
        })
    
    def test_process_weather_features(self):
        """Test complete weather feature processing"""
        result = self.engine.process_weather_features(self.sample_data)
        
        # Check that all stress columns are added
        expected_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that values are in valid range [0, 1]
        validation = self.engine.validate_stress_scores(result)
        for col, is_valid in validation.items():
            self.assertTrue(is_valid, f"{col} has invalid values")
    
    def test_validate_stress_scores(self):
        """Test stress score validation"""
        # Create data with valid scores
        valid_data = pd.DataFrame({
            'thermal_stress': [0.0, 0.5, 1.0],
            'wind_stress': [0.2, 0.7, 0.9],
            'precip_stress': [0.0, 0.3, 1.0],
            'storm_proxy': [0.1, 0.6, 0.8]
        })
        
        validation = self.engine.validate_stress_scores(valid_data)
        for col, is_valid in validation.items():
            self.assertTrue(is_valid)
        
        # Create data with invalid scores
        invalid_data = pd.DataFrame({
            'thermal_stress': [-0.1, 0.5, 1.2],  # Invalid range
            'wind_stress': [0.2, 0.7, 0.9],
            'precip_stress': [0.0, 0.3, 1.0],
            'storm_proxy': [0.1, 0.6, 0.8]
        })
        
        validation_invalid = self.engine.validate_stress_scores(invalid_data)
        self.assertFalse(validation_invalid['thermal_stress'])
    
    def test_get_feature_summary(self):
        """Test feature summary statistics"""
        processed_data = self.engine.process_weather_features(self.sample_data)
        summary = self.engine.get_feature_summary(processed_data)
        
        # Check that summary contains expected statistics
        expected_stats = ['mean', 'std', 'min', 'max', 'median', 'q75', 'q95']
        for stress_type in summary:
            for stat in expected_stats:
                self.assertIn(stat, summary[stress_type])
                self.assertIsInstance(summary[stress_type][stat], float)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility conversion functions"""
    
    def test_temperature_conversions(self):
        """Test temperature conversion functions"""
        # Celsius to Fahrenheit
        self.assertAlmostEqual(celsius_to_fahrenheit(0.0), 32.0, places=1)
        self.assertAlmostEqual(celsius_to_fahrenheit(100.0), 212.0, places=1)
        self.assertAlmostEqual(celsius_to_fahrenheit(37.0), 98.6, places=1)
        
        # Fahrenheit to Celsius
        self.assertAlmostEqual(fahrenheit_to_celsius(32.0), 0.0, places=1)
        self.assertAlmostEqual(fahrenheit_to_celsius(212.0), 100.0, places=1)
        self.assertAlmostEqual(fahrenheit_to_celsius(98.6), 37.0, places=1)
        
        # Round trip conversion
        temp_c = 25.0
        temp_f = celsius_to_fahrenheit(temp_c)
        temp_c_back = fahrenheit_to_celsius(temp_f)
        self.assertAlmostEqual(temp_c, temp_c_back, places=3)
    
    def test_wind_speed_conversions(self):
        """Test wind speed conversion functions"""
        # mph to m/s
        self.assertAlmostEqual(mph_to_ms(0.0), 0.0, places=3)
        self.assertAlmostEqual(mph_to_ms(22.369), 10.0, places=2)  # Approximate
        
        # m/s to mph
        self.assertAlmostEqual(ms_to_mph(0.0), 0.0, places=3)
        self.assertAlmostEqual(ms_to_mph(10.0), 22.369, places=2)  # Approximate
        
        # Round trip conversion
        speed_mph = 35.0
        speed_ms = mph_to_ms(speed_mph)
        speed_mph_back = ms_to_mph(speed_ms)
        self.assertAlmostEqual(speed_mph, speed_mph_back, places=3)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)