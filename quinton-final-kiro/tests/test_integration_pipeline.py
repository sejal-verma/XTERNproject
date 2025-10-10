"""
Integration Tests for Complete Pipeline Execution

This module provides comprehensive integration tests that validate the complete
MISO Weather-Stress Heatmap pipeline from data ingestion through final output
generation, including extended components and demo data scenarios.

Requirements addressed: 5.1, 7.4
"""

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import json

from demo_data_generator import DemoDataGenerator, validate_demo_data_quality
from extensibility_framework import ExtensibilityManager, ComponentConfig, ComponentType
from extended_risk_integration import ExtendedRiskScoringEngine
from extension_examples import create_sample_extension_data
from feature_engineering import FeatureEngineeringEngine
from risk_scoring_engine import RiskScoringEngine


class TestDemoDataGeneration(unittest.TestCase):
    """Test demo data generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DemoDataGenerator(random_seed=42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hex_grid_generation(self):
        """Test hexagonal grid generation"""
        grid = self.generator.generate_hex_grid_demo(n_cells=50)
        
        # Check basic properties
        self.assertEqual(len(grid), 50)
        self.assertIn('cell_id', grid.columns)
        self.assertIn('geometry', grid.columns)
        self.assertIn('centroid_lat', grid.columns)
        self.assertIn('centroid_lon', grid.columns)
        
        # Check coordinate ranges
        self.assertTrue((grid['centroid_lat'] >= 37.0).all())
        self.assertTrue((grid['centroid_lat'] <= 49.0).all())
        self.assertTrue((grid['centroid_lon'] >= -104.0).all())
        self.assertTrue((grid['centroid_lon'] <= -82.0).all())
        
        # Check unique cell IDs
        self.assertEqual(len(grid['cell_id'].unique()), 50)
    
    def test_weather_data_generation(self):
        """Test weather data generation"""
        grid = self.generator.generate_hex_grid_demo(n_cells=10)
        weather_data = self.generator.generate_weather_demo_data(
            grid, horizons_h=[12, 24], scenario='normal'
        )
        
        # Check basic properties
        expected_records = 10 * 2  # 10 cells × 2 horizons
        self.assertEqual(len(weather_data), expected_records)
        
        # Check required columns
        required_columns = [
            'cell_id', 'horizon_h', 'timestamp', 'temp_2m', 'heat_index',
            'wind_speed', 'wind_gust', 'precip_rate', 'snow_rate', 'ice_rate'
        ]
        for col in required_columns:
            self.assertIn(col, weather_data.columns)
        
        # Check data ranges
        self.assertTrue((weather_data['temp_2m'] >= -50).all())
        self.assertTrue((weather_data['temp_2m'] <= 130).all())
        self.assertTrue((weather_data['heat_index'] >= weather_data['temp_2m']).all())
        self.assertTrue((weather_data['wind_gust'] >= weather_data['wind_speed']).all())
        self.assertTrue((weather_data['precip_rate'] >= 0).all())
    
    def test_infrastructure_data_generation(self):
        """Test infrastructure data generation"""
        grid = self.generator.generate_hex_grid_demo(n_cells=10)
        infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
        
        # Check basic properties
        self.assertEqual(len(infrastructure_data), 10)
        
        # Check required columns
        required_columns = [
            'cell_id', 'population_density', 'normalized_pop_density',
            'renewable_share', 'transmission_scarcity'
        ]
        for col in required_columns:
            self.assertIn(col, infrastructure_data.columns)
        
        # Check data ranges
        self.assertTrue((infrastructure_data['population_density'] > 0).all())
        self.assertTrue((infrastructure_data['normalized_pop_density'] >= 0).all())
        self.assertTrue((infrastructure_data['normalized_pop_density'] <= 1).all())
        self.assertTrue((infrastructure_data['renewable_share'] >= 0).all())
        self.assertTrue((infrastructure_data['renewable_share'] <= 1).all())
    
    def test_scenario_variations(self):
        """Test different weather scenarios"""
        grid = self.generator.generate_hex_grid_demo(n_cells=5)
        
        scenarios = ['normal', 'heat_wave', 'winter_storm', 'severe_weather']
        scenario_data = {}
        
        for scenario in scenarios:
            weather_data = self.generator.generate_weather_demo_data(
                grid, horizons_h=[24], scenario=scenario
            )
            scenario_data[scenario] = weather_data
        
        # Heat wave should have higher temperatures
        normal_temp = scenario_data['normal']['temp_2m'].mean()
        heat_wave_temp = scenario_data['heat_wave']['temp_2m'].mean()
        self.assertGreater(heat_wave_temp, normal_temp)
        
        # Winter storm should have lower temperatures
        winter_temp = scenario_data['winter_storm']['temp_2m'].mean()
        self.assertLess(winter_temp, normal_temp)
        
        # Severe weather should have higher wind speeds
        normal_wind = scenario_data['normal']['wind_gust'].mean()
        severe_wind = scenario_data['severe_weather']['wind_gust'].mean()
        self.assertGreater(severe_wind, normal_wind)
    
    def test_save_demo_datasets(self):
        """Test saving complete demo datasets"""
        saved_files = self.generator.save_demo_datasets(
            output_dir=self.temp_dir,
            scenarios=['normal', 'heat_wave']
        )
        
        # Check that all expected files were created
        self.assertIn('grid', saved_files)
        self.assertIn('infrastructure', saved_files)
        self.assertIn('weather_scenarios', saved_files)
        self.assertIn('metadata', saved_files)
        
        # Check file existence
        self.assertTrue(os.path.exists(saved_files['grid']))
        self.assertTrue(os.path.exists(saved_files['infrastructure']))
        self.assertTrue(os.path.exists(saved_files['metadata']))
        
        for scenario_file in saved_files['weather_scenarios'].values():
            self.assertTrue(os.path.exists(scenario_file))
        
        # Check metadata content
        with open(saved_files['metadata'], 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('generated_at', metadata)
        self.assertIn('random_seed', metadata)
        self.assertIn('scenarios', metadata)
        self.assertEqual(metadata['scenarios'], ['normal', 'heat_wave'])
    
    def test_demo_data_quality_validation(self):
        """Test demo data quality validation"""
        grid = self.generator.generate_hex_grid_demo(n_cells=20)
        weather_data = self.generator.generate_weather_demo_data(grid)
        infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
        
        validation_results = validate_demo_data_quality(weather_data, infrastructure_data)
        
        # Check that validation passes
        self.assertTrue(validation_results['overall_quality'])
        
        # Check specific validations
        weather_val = validation_results['weather_validation']
        self.assertTrue(weather_val['temperature_range_realistic'])
        self.assertTrue(weather_val['heat_index_valid'])
        self.assertTrue(weather_val['wind_gust_higher'])
        self.assertTrue(weather_val['precipitation_non_negative'])
        
        infra_val = validation_results['infrastructure_validation']
        self.assertTrue(infra_val['renewable_share_valid'])
        self.assertTrue(infra_val['population_positive'])
        self.assertTrue(infra_val['capacity_consistent'])


class TestCompleteIntegrationPipeline(unittest.TestCase):
    """Test complete integration pipeline with real components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DemoDataGenerator(random_seed=42)
        
        # Generate demo data
        self.grid = self.generator.generate_hex_grid_demo(n_cells=20)
        self.weather_data = self.generator.generate_weather_demo_data(self.grid)
        self.infrastructure_data = self.generator.generate_infrastructure_demo_data(self.grid)
        
        # Initialize engines
        self.feature_engine = FeatureEngineeringEngine()
        self.risk_engine = RiskScoringEngine()
        
        # Extended system configuration
        self.extended_config = {
            'weights': {
                'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15,
                'pop': 0.7, 'load': 0.3,
                'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1,
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            },
            'extended_components': {
                'resource_transition': {
                    'type': 'vulnerability',
                    'weight': 0.3,
                    'enabled': True,
                    'parameters': {}
                },
                'load_growth': {
                    'type': 'exposure',
                    'weight': 0.4,
                    'enabled': True,
                    'parameters': {}
                }
            }
        }
        
        self.extended_engine = ExtendedRiskScoringEngine(self.extended_config)
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline"""
        # Process weather features
        processed_weather = self.feature_engine.process_weather_features(self.weather_data)
        
        # Check that stress scores were added
        stress_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        for col in stress_columns:
            self.assertIn(col, processed_weather.columns)
        
        # Validate stress scores
        validation_results = self.feature_engine.validate_stress_scores(processed_weather)
        for col in stress_columns:
            self.assertTrue(validation_results[col], f"Invalid {col} scores")
        
        # Check score ranges
        for col in stress_columns:
            scores = processed_weather[col]
            self.assertTrue((scores >= 0).all(), f"{col} has negative values")
            self.assertTrue((scores <= 1).all(), f"{col} has values > 1")
    
    def test_core_risk_scoring_pipeline(self):
        """Test core risk scoring pipeline"""
        # Process weather features first
        processed_weather = self.feature_engine.process_weather_features(self.weather_data)
        
        # Calculate hazard scores
        hazard_data = self.risk_engine.process_hazard_scores(processed_weather)
        self.assertIn('hazard_score', hazard_data.columns)
        
        # Calculate exposure scores
        exposure_data = self.risk_engine.process_exposure_scores(self.infrastructure_data)
        self.assertIn('exposure_score', exposure_data.columns)
        
        # Calculate vulnerability scores
        vulnerability_data = self.risk_engine.process_vulnerability_scores(self.infrastructure_data)
        self.assertIn('vulnerability_score', vulnerability_data.columns)
        
        # Validate all scores are in [0, 1] range
        for score_col in ['hazard_score', 'exposure_score', 'vulnerability_score']:
            if score_col == 'hazard_score':
                scores = hazard_data[score_col]
            elif score_col == 'exposure_score':
                scores = exposure_data[score_col]
            else:
                scores = vulnerability_data[score_col]
            
            self.assertTrue((scores >= 0).all(), f"{score_col} has negative values")
            self.assertTrue((scores <= 1).all(), f"{score_col} has values > 1")
    
    def test_extended_risk_scoring_pipeline(self):
        """Test extended risk scoring pipeline with plugins"""
        # Add extended data columns for components
        extended_infrastructure = self.infrastructure_data.copy()
        extended_infrastructure['renewable_transition_rate'] = np.random.uniform(0.02, 0.12, len(extended_infrastructure))
        extended_infrastructure['projected_load_growth_rate'] = np.random.uniform(0.01, 0.08, len(extended_infrastructure))
        
        # Process weather features
        processed_weather = self.feature_engine.process_weather_features(self.weather_data)
        
        # Calculate extended scores
        extended_hazard = self.extended_engine.calculate_extended_hazard_score(processed_weather)
        extended_exposure = self.extended_engine.calculate_extended_exposure_score(extended_infrastructure)
        extended_vulnerability = self.extended_engine.calculate_extended_vulnerability_score(extended_infrastructure)
        
        # Check that extended scores were calculated
        self.assertIn('extended_hazard_score', extended_hazard.columns)
        self.assertIn('extended_exposure_score', extended_exposure.columns)
        self.assertIn('extended_vulnerability_score', extended_vulnerability.columns)
        
        # Validate score ranges
        for df, col in [(extended_hazard, 'extended_hazard_score'),
                       (extended_exposure, 'extended_exposure_score'),
                       (extended_vulnerability, 'extended_vulnerability_score')]:
            scores = df[col]
            self.assertTrue((scores >= 0).all(), f"{col} has negative values")
            self.assertTrue((scores <= 1).all(), f"{col} has values > 1")
    
    def test_end_to_end_pipeline_execution(self):
        """Test complete end-to-end pipeline execution"""
        # Add extended data for complete test
        extended_infrastructure = self.infrastructure_data.copy()
        extended_infrastructure['renewable_transition_rate'] = np.random.uniform(0.02, 0.12, len(extended_infrastructure))
        extended_infrastructure['projected_load_growth_rate'] = np.random.uniform(0.01, 0.08, len(extended_infrastructure))
        
        # Step 1: Feature Engineering
        processed_weather = self.feature_engine.process_weather_features(self.weather_data)
        
        # Step 2: Extended Risk Scoring
        extended_hazard = self.extended_engine.calculate_extended_hazard_score(processed_weather)
        extended_exposure = self.extended_engine.calculate_extended_exposure_score(extended_infrastructure)
        extended_vulnerability = self.extended_engine.calculate_extended_vulnerability_score(extended_infrastructure)
        
        # Step 3: Validate System
        # First ensure we have the required columns for validation
        if 'hazard_score' not in extended_hazard.columns:
            # Add hazard score for validation
            hazard_with_score = self.risk_engine.process_hazard_scores(processed_weather)
            validation_weather = hazard_with_score
        else:
            validation_weather = extended_hazard
            
        if 'exposure_score' not in extended_infrastructure.columns:
            # Add exposure score for validation
            exposure_with_score = self.risk_engine.process_exposure_scores(extended_infrastructure)
            validation_infra = exposure_with_score
        else:
            validation_infra = extended_infrastructure
            
        validation_results = self.extended_engine.validate_extended_system(
            validation_weather, validation_infra
        )
        
        # Check validation results
        self.assertTrue(validation_results['core_system_valid'])
        self.assertTrue(validation_results['extended_components_valid'])
        self.assertTrue(validation_results['integration_valid'])
        
        # Step 4: Check data consistency across pipeline
        # All dataframes should have same number of unique cell_ids
        weather_cells = set(processed_weather['cell_id'].unique())
        infra_cells = set(extended_infrastructure['cell_id'].unique())
        
        self.assertEqual(weather_cells, infra_cells, "Cell IDs don't match between datasets")
        
        # Check that we have data for all forecast horizons
        horizons = processed_weather['horizon_h'].unique()
        self.assertGreater(len(horizons), 0, "No forecast horizons found")
        
        # Verify score distributions are reasonable
        hazard_mean = extended_hazard['extended_hazard_score'].mean()
        exposure_mean = extended_exposure['extended_exposure_score'].mean()
        vulnerability_mean = extended_vulnerability['extended_vulnerability_score'].mean()
        
        # Scores should be distributed across the range, not all at extremes
        self.assertGreater(hazard_mean, 0.05, "Hazard scores too low")
        self.assertLess(hazard_mean, 0.95, "Hazard scores too high")
        self.assertGreater(exposure_mean, 0.05, "Exposure scores too low")
        self.assertLess(exposure_mean, 0.95, "Exposure scores too high")
        self.assertGreater(vulnerability_mean, 0.05, "Vulnerability scores too low")
        self.assertLess(vulnerability_mean, 0.95, "Vulnerability scores too high")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks and scalability"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DemoDataGenerator(random_seed=42)
    
    def test_performance_scaling(self):
        """Test performance with different data sizes"""
        import time
        
        grid_sizes = [50, 100, 200]
        performance_results = {}
        
        for n_cells in grid_sizes:
            start_time = time.time()
            
            # Generate data
            grid = self.generator.generate_hex_grid_demo(n_cells=n_cells)
            weather_data = self.generator.generate_weather_demo_data(grid, horizons_h=[12, 24])
            infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
            
            # Process through pipeline
            feature_engine = FeatureEngineeringEngine()
            processed_weather = feature_engine.process_weather_features(weather_data)
            
            risk_engine = RiskScoringEngine()
            hazard_data = risk_engine.process_hazard_scores(processed_weather)
            exposure_data = risk_engine.process_exposure_scores(infrastructure_data)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            performance_results[n_cells] = {
                'processing_time': processing_time,
                'records_per_second': len(processed_weather) / processing_time,
                'weather_records': len(processed_weather),
                'infrastructure_records': len(infrastructure_data)
            }
        
        # Check that performance scales reasonably
        # Processing time should increase with data size, but not exponentially
        times = [performance_results[n]['processing_time'] for n in grid_sizes]
        
        # Larger datasets should take more time, but not too much more
        self.assertGreater(times[1], times[0], "Processing time should increase with data size")
        self.assertGreater(times[2], times[1], "Processing time should increase with data size")
        
        # But it shouldn't be exponential (rough check)
        time_ratio_1 = times[1] / times[0]
        time_ratio_2 = times[2] / times[1]
        size_ratio = 2.0  # Each step doubles the size
        
        # Time ratio should be less than size_ratio^2 (quadratic would be bad)
        self.assertLess(time_ratio_1, size_ratio ** 1.5, "Performance scaling too poor")
        self.assertLess(time_ratio_2, size_ratio ** 1.5, "Performance scaling too poor")
    
    def test_memory_usage_validation(self):
        """Test that memory usage remains reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a moderately large dataset
        grid = self.generator.generate_hex_grid_demo(n_cells=500)
        weather_data = self.generator.generate_weather_demo_data(grid)
        infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
        
        # Process through complete pipeline
        feature_engine = FeatureEngineeringEngine()
        processed_weather = feature_engine.process_weather_features(weather_data)
        
        extended_config = {
            'weights': {
                'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15,
                'pop': 0.7, 'load': 0.3,
                'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1,
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            }
        }
        
        extended_engine = ExtendedRiskScoringEngine(extended_config)
        hazard_result = extended_engine.calculate_extended_hazard_score(processed_weather)
        exposure_result = extended_engine.calculate_extended_exposure_score(infrastructure_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500, f"Memory usage too high: {memory_increase:.1f} MB")


class TestErrorHandlingAndRobustness(unittest.TestCase):
    """Test error handling and system robustness"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DemoDataGenerator(random_seed=42)
        self.feature_engine = FeatureEngineeringEngine()
        self.risk_engine = RiskScoringEngine()
    
    def test_missing_data_handling(self):
        """Test handling of missing data columns"""
        # Create incomplete weather data
        incomplete_weather = pd.DataFrame({
            'temp_2m': [70, 80, 90],
            'wind_speed': [10, 15, 20]
            # Missing heat_index, wind_gust, etc.
        })
        
        # Feature engineering should handle missing columns gracefully
        processed = self.feature_engine.process_weather_features(incomplete_weather)
        
        # Should not raise an exception and should have stress score columns
        self.assertIsInstance(processed, pd.DataFrame)
        stress_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        for col in stress_columns:
            self.assertIn(col, processed.columns)
        
        # Should have filled missing columns with defaults
        self.assertIn('heat_index', processed.columns)
        self.assertIn('wind_gust', processed.columns)
    
    def test_invalid_data_ranges(self):
        """Test handling of invalid data ranges"""
        # Create weather data with extreme values
        extreme_weather = pd.DataFrame({
            'temp_2m': [-100, 200, np.nan],  # Extreme and NaN values
            'heat_index': [-50, 250, np.inf],  # Extreme and inf values
            'wind_speed': [-10, 1000, np.nan],  # Negative and extreme values
            'wind_gust': [0, 2000, np.nan],
            'precip_rate': [-5, 1000, np.nan],
            'snow_rate': [-2, 500, np.nan],
            'ice_rate': [-1, 100, np.nan]
        })
        
        # System should handle extreme values without crashing
        try:
            processed = self.feature_engine.process_weather_features(extreme_weather)
            
            # Check that stress scores are still in valid range
            stress_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
            for col in stress_columns:
                if col in processed.columns:
                    scores = processed[col]
                    # Remove NaN values for testing
                    valid_scores = scores.dropna()
                    if len(valid_scores) > 0:
                        self.assertTrue((valid_scores >= 0).all(), f"{col} has negative values")
                        self.assertTrue((valid_scores <= 1).all(), f"{col} has values > 1")
        
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_weather = pd.DataFrame()
        empty_infrastructure = pd.DataFrame()
        
        # System should handle empty dataframes without crashing
        processed_weather = self.feature_engine.process_weather_features(empty_weather)
        self.assertIsInstance(processed_weather, pd.DataFrame)
        # Should have expected columns even if empty
        expected_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        for col in expected_columns:
            self.assertIn(col, processed_weather.columns)
        
        exposure_result = self.risk_engine.process_exposure_scores(empty_infrastructure)
        self.assertIsInstance(exposure_result, pd.DataFrame)
        # Should have expected columns even if empty
        self.assertIn('exposure_score', exposure_result.columns)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests with high verbosity
    unittest.main(verbosity=2)