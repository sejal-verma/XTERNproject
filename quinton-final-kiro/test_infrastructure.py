# Test Infrastructure Data Processing
# Validation for Task 4: Infrastructure and Exposure Data System

import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infrastructure_adapters import (
    GenerationCapacityProcessor,
    PopulationExposureProcessor, 
    TransmissionDensityProcessor,
    InfrastructureDataSystem
)
from spatial_framework import SpatialProcessingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class InfrastructureDataTests:
    """Test suite for infrastructure data processing"""
    
    def __init__(self):
        self.test_results = {}
        self.config = self._create_test_config()
        self.spatial_engine = None
        self.grid = None
        
    def _create_test_config(self) -> dict:
        """Create test configuration"""
        return {
            'runtime': {
                'mode': 'demo',
                'crs': 'EPSG:4326',
                'hex_size_km': 40
            },
            'weights': {
                'exposure': {
                    'pop': 0.7,
                    'load': 0.3
                },
                'vulnerability': {
                    'renew_share': 0.6,
                    'tx_scarcity': 0.3,
                    'outage': 0.1
                }
            }
        }
    
    def setup_test_grid(self) -> gpd.GeoDataFrame:
        """Create test grid for infrastructure processing"""
        logging.info("Setting up test grid...")
        
        # Initialize spatial engine
        self.spatial_engine = SpatialProcessingEngine(self.config)
        footprint, grid = self.spatial_engine.initialize_spatial_framework()
        
        self.grid = grid
        logging.info(f"Test grid created: {len(grid)} cells")
        return grid
    
    def run_all_tests(self) -> dict:
        """Run all infrastructure data tests"""
        logging.info("Running infrastructure data tests...")
        
        # Setup test grid
        if self.grid is None:
            self.setup_test_grid()
        
        tests = [
            ('test_capacity_processor_initialization', self.test_capacity_processor_initialization),
            ('test_capacity_data_loading', self.test_capacity_data_loading),
            ('test_capacity_grid_processing', self.test_capacity_grid_processing),
            ('test_population_processor_initialization', self.test_population_processor_initialization),
            ('test_population_data_loading', self.test_population_data_loading),
            ('test_population_grid_processing', self.test_population_grid_processing),
            ('test_transmission_processor_initialization', self.test_transmission_processor_initialization),
            ('test_transmission_data_loading', self.test_transmission_data_loading),
            ('test_transmission_grid_processing', self.test_transmission_grid_processing),
            ('test_infrastructure_system_integration', self.test_infrastructure_system_integration),
            ('test_data_validation', self.test_data_validation),
            ('test_renewable_share_calculation', self.test_renewable_share_calculation),
            ('test_population_normalization', self.test_population_normalization),
            ('test_transmission_scarcity_calculation', self.test_transmission_scarcity_calculation)
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
        logging.info(f"Infrastructure Test Results: {passed}/{total} passed")
        
        return self.test_results
    
    def test_capacity_processor_initialization(self) -> bool:
        """Test generation capacity processor initialization"""
        try:
            processor = GenerationCapacityProcessor(self.config, demo_mode=True)
            
            # Check initialization
            if processor.config != self.config:
                return False
            
            if not processor.demo_mode:
                return False
            
            if processor.capacity_data is not None:
                return False  # Should be None before loading
            
            return True
            
        except Exception as e:
            logging.error(f"Capacity processor initialization failed: {e}")
            return False
    
    def test_capacity_data_loading(self) -> bool:
        """Test generation capacity data loading"""
        try:
            processor = GenerationCapacityProcessor(self.config, demo_mode=True)
            capacity_data = processor.load_data()
            
            # Check data structure
            if not isinstance(capacity_data, gpd.GeoDataFrame):
                return False
            
            # Check required columns
            required_cols = ['facility_id', 'fuel_type', 'capacity_mw', 'longitude', 'latitude']
            if not all(col in capacity_data.columns for col in required_cols):
                return False
            
            # Check data content
            if len(capacity_data) == 0:
                return False
            
            # Check geometry
            if not hasattr(capacity_data, 'geometry'):
                return False
            
            # Check fuel types are realistic
            fuel_types = capacity_data['fuel_type'].unique()
            expected_fuels = ['Wind', 'Solar Photovoltaic', 'Natural Gas', 'Coal', 'Nuclear']
            if not any(fuel in fuel_types for fuel in expected_fuels):
                return False
            
            # Check capacity values are positive
            if not (capacity_data['capacity_mw'] > 0).all():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Capacity data loading failed: {e}")
            return False
    
    def test_capacity_grid_processing(self) -> bool:
        """Test capacity data processing to grid"""
        try:
            processor = GenerationCapacityProcessor(self.config, demo_mode=True)
            processor.load_data()
            
            result = processor.process_to_grid(self.grid)
            
            # Check result structure
            if not isinstance(result, pd.DataFrame):
                return False
            
            # Check required columns
            required_cols = [
                'cell_id', 'total_capacity_mw', 'renewable_capacity_mw', 
                'renewable_share', 'capacity_density_mw_km2'
            ]
            if not all(col in result.columns for col in required_cols):
                return False
            
            # Check all grid cells are represented
            if len(result) != len(self.grid):
                return False
            
            # Check cell_id matching
            if not set(result['cell_id']) == set(self.grid['cell_id']):
                return False
            
            # Check renewable share is between 0 and 1
            if not result['renewable_share'].between(0, 1).all():
                return False
            
            # Check capacity values are non-negative
            if not (result['total_capacity_mw'] >= 0).all():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Capacity grid processing failed: {e}")
            return False
    
    def test_population_processor_initialization(self) -> bool:
        """Test population processor initialization"""
        try:
            processor = PopulationExposureProcessor(self.config, demo_mode=True)
            
            # Check initialization
            if processor.config != self.config:
                return False
            
            if not processor.demo_mode:
                return False
            
            if processor.population_data is not None:
                return False  # Should be None before loading
            
            return True
            
        except Exception as e:
            logging.error(f"Population processor initialization failed: {e}")
            return False
    
    def test_population_data_loading(self) -> bool:
        """Test population data loading"""
        try:
            processor = PopulationExposureProcessor(self.config, demo_mode=True)
            population_data = processor.load_data()
            
            # Check data structure
            if not isinstance(population_data, gpd.GeoDataFrame):
                return False
            
            # Check required columns
            required_cols = ['area_id', 'population', 'area_km2', 'population_density_per_km2']
            if not all(col in population_data.columns for col in required_cols):
                return False
            
            # Check data content
            if len(population_data) == 0:
                return False
            
            # Check population values are positive
            if not (population_data['population'] > 0).all():
                return False
            
            # Check density calculation consistency
            calculated_density = population_data['population'] / population_data['area_km2']
            density_diff = abs(calculated_density - population_data['population_density_per_km2'])
            if not (density_diff < 1e-6).all():  # Allow for small floating point errors
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Population data loading failed: {e}")
            return False
    
    def test_population_grid_processing(self) -> bool:
        """Test population data processing to grid"""
        try:
            processor = PopulationExposureProcessor(self.config, demo_mode=True)
            processor.load_data()
            
            result = processor.process_to_grid(self.grid)
            
            # Check result structure
            if not isinstance(result, pd.DataFrame):
                return False
            
            # Check required columns
            required_cols = [
                'cell_id', 'total_population', 'population_density_per_km2', 
                'normalized_pop_density', 'load_factor', 'exposure_score'
            ]
            if not all(col in result.columns for col in required_cols):
                return False
            
            # Check all grid cells are represented
            if len(result) != len(self.grid):
                return False
            
            # Check normalized values are between 0 and 1
            if not result['normalized_pop_density'].between(0, 1).all():
                return False
            
            if not result['load_factor'].between(0, 1).all():
                return False
            
            if not result['exposure_score'].between(0, 1).all():
                return False
            
            # Check population values are non-negative
            if not (result['total_population'] >= 0).all():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Population grid processing failed: {e}")
            return False
    
    def test_transmission_processor_initialization(self) -> bool:
        """Test transmission processor initialization"""
        try:
            processor = TransmissionDensityProcessor(self.config, demo_mode=True)
            
            # Check initialization
            if processor.config != self.config:
                return False
            
            if not processor.demo_mode:
                return False
            
            if processor.baseline_scarcity != 0.5:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Transmission processor initialization failed: {e}")
            return False
    
    def test_transmission_data_loading(self) -> bool:
        """Test transmission data loading"""
        try:
            processor = TransmissionDensityProcessor(self.config, demo_mode=True)
            transmission_data = processor.load_data()
            
            # In demo mode, should return data
            if transmission_data is None:
                return False
            
            # Check data structure
            if not isinstance(transmission_data, gpd.GeoDataFrame):
                return False
            
            # Check required columns
            required_cols = ['line_id', 'voltage_kv', 'line_type']
            if not all(col in transmission_data.columns for col in required_cols):
                return False
            
            # Check geometry type (should be LineString)
            geom_types = transmission_data.geometry.geom_type.unique()
            if 'LineString' not in geom_types:
                return False
            
            # Check voltage values are realistic
            voltages = transmission_data['voltage_kv'].unique()
            if not any(v >= 100 for v in voltages):  # Should have some high voltage lines
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Transmission data loading failed: {e}")
            return False
    
    def test_transmission_grid_processing(self) -> bool:
        """Test transmission data processing to grid"""
        try:
            processor = TransmissionDensityProcessor(self.config, demo_mode=True)
            processor.load_data()
            
            result = processor.process_to_grid(self.grid)
            
            # Check result structure
            if not isinstance(result, pd.DataFrame):
                return False
            
            # Check required columns
            required_cols = [
                'cell_id', 'transmission_line_count', 'transmission_density_km_per_km2',
                'transmission_scarcity', 'data_source'
            ]
            if not all(col in result.columns for col in required_cols):
                return False
            
            # Check all grid cells are represented
            if len(result) != len(self.grid):
                return False
            
            # Check scarcity values are between 0 and 1
            if not result['transmission_scarcity'].between(0, 1).all():
                return False
            
            # Check line counts are non-negative integers
            if not (result['transmission_line_count'] >= 0).all():
                return False
            
            # Check density values are non-negative
            if not (result['transmission_density_km_per_km2'] >= 0).all():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Transmission grid processing failed: {e}")
            return False
    
    def test_infrastructure_system_integration(self) -> bool:
        """Test complete infrastructure system integration"""
        try:
            system = InfrastructureDataSystem(self.config)
            combined_data = system.initialize_infrastructure_data(self.grid)
            
            # Check result structure
            if not isinstance(combined_data, pd.DataFrame):
                return False
            
            # Check all grid cells are represented
            if len(combined_data) != len(self.grid):
                return False
            
            # Check key columns from each processor are present
            capacity_cols = ['total_capacity_mw', 'renewable_share']
            population_cols = ['total_population', 'exposure_score']
            transmission_cols = ['transmission_scarcity']
            
            all_required_cols = capacity_cols + population_cols + transmission_cols
            if not all(col in combined_data.columns for col in all_required_cols):
                return False
            
            # Check cell_id matching
            if not set(combined_data['cell_id']) == set(self.grid['cell_id']):
                return False
            
            # Check no missing values in key columns
            if combined_data[all_required_cols].isna().any().any():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Infrastructure system integration failed: {e}")
            return False
    
    def test_data_validation(self) -> bool:
        """Test data validation and quality checks"""
        try:
            system = InfrastructureDataSystem(self.config)
            combined_data = system.initialize_infrastructure_data(self.grid)
            
            # Check value ranges
            # Renewable share should be [0,1]
            if not combined_data['renewable_share'].between(0, 1).all():
                return False
            
            # Exposure score should be [0,1]
            if not combined_data['exposure_score'].between(0, 1).all():
                return False
            
            # Transmission scarcity should be [0,1]
            if not combined_data['transmission_scarcity'].between(0, 1).all():
                return False
            
            # Capacity values should be non-negative
            if not (combined_data['total_capacity_mw'] >= 0).all():
                return False
            
            # Population should be non-negative
            if not (combined_data['total_population'] >= 0).all():
                return False
            
            # Check for reasonable data distribution
            # Should have some variation in values (not all zeros or all same)
            if combined_data['renewable_share'].nunique() < 2:
                return False
            
            if combined_data['exposure_score'].nunique() < 2:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return False
    
    def test_renewable_share_calculation(self) -> bool:
        """Test renewable share calculation accuracy"""
        try:
            processor = GenerationCapacityProcessor(self.config, demo_mode=True)
            processor.load_data()
            result = processor.process_to_grid(self.grid)
            
            # Check renewable share calculation
            for _, row in result.iterrows():
                total_cap = row['total_capacity_mw']
                renewable_cap = row['renewable_capacity_mw']
                renewable_share = row['renewable_share']
                
                if total_cap > 0:
                    expected_share = renewable_cap / total_cap
                    if abs(renewable_share - expected_share) > 1e-6:
                        return False
                else:
                    # No capacity should mean 0 renewable share
                    if renewable_share != 0.0:
                        return False
            
            return True
            
        except Exception as e:
            logging.error(f"Renewable share calculation test failed: {e}")
            return False
    
    def test_population_normalization(self) -> bool:
        """Test population density normalization"""
        try:
            processor = PopulationExposureProcessor(self.config, demo_mode=True)
            processor.load_data()
            result = processor.process_to_grid(self.grid)
            
            # Check normalization
            max_density = result['population_density_per_km2'].max()
            max_normalized = result['normalized_pop_density'].max()
            
            if max_density > 0:
                # Maximum normalized value should be 1.0
                if abs(max_normalized - 1.0) > 1e-6:
                    return False
                
                # Check proportional scaling
                for _, row in result.iterrows():
                    density = row['population_density_per_km2']
                    normalized = row['normalized_pop_density']
                    expected_normalized = density / max_density
                    
                    if abs(normalized - expected_normalized) > 1e-6:
                        return False
            
            return True
            
        except Exception as e:
            logging.error(f"Population normalization test failed: {e}")
            return False
    
    def test_transmission_scarcity_calculation(self) -> bool:
        """Test transmission scarcity calculation"""
        try:
            processor = TransmissionDensityProcessor(self.config, demo_mode=True)
            processor.load_data()
            result = processor.process_to_grid(self.grid)
            
            # Check scarcity calculation logic
            for _, row in result.iterrows():
                density = row['transmission_density_km_per_km2']
                scarcity = row['transmission_scarcity']
                
                # Higher density should mean lower scarcity
                # Scarcity should be between 0 and 1
                if not (0 <= scarcity <= 1):
                    return False
                
                # If no transmission lines, scarcity should be high
                if row['transmission_line_count'] == 0:
                    if scarcity < 0.5:  # Should be relatively high scarcity
                        return False
            
            return True
            
        except Exception as e:
            logging.error(f"Transmission scarcity calculation test failed: {e}")
            return False
    
    def get_test_summary(self) -> str:
        """Get formatted test summary"""
        if not self.test_results:
            return "No tests run yet"
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        summary = f"\n=== Infrastructure Data Test Summary ===\n"
        summary += f"Tests passed: {passed}/{total}\n\n"
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            summary += f"{status}: {test_name}\n"
        
        return summary


def main():
    """Run infrastructure data tests"""
    print("Testing Infrastructure Data Processing System...")
    print("=" * 50)
    
    # Create test instance
    tester = InfrastructureDataTests()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    print(tester.get_test_summary())
    
    # Return success/failure
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All infrastructure data tests passed!")
        return 0
    else:
        print("\n❌ Some infrastructure data tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())