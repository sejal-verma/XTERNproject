"""
Unit Tests for Extensibility Framework

This module provides comprehensive unit tests for the extensibility framework,
including plugin architecture, component registration, and integration testing.

Requirements addressed: 5.1, 7.4
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
import json

from extensibility_framework import (
    ExtensibilityManager, BaseRiskComponent, ComponentConfig, ComponentType,
    ResourceTransitionComponent, LoadGrowthComponent
)
from extended_risk_integration import ExtendedRiskScoringEngine
from extension_examples import (
    CyberSecurityRiskComponent, MarketVolatilityComponent,
    SeasonalDemandComponent, InfrastructureAgeComponent,
    create_sample_extension_data
)


class TestComponentConfig(unittest.TestCase):
    """Test ComponentConfig dataclass"""
    
    def test_component_config_creation(self):
        """Test creating component configuration"""
        config = ComponentConfig(
            name='test_component',
            component_type=ComponentType.HAZARD,
            weight=0.3,
            enabled=True,
            parameters={'param1': 'value1'}
        )
        
        self.assertEqual(config.name, 'test_component')
        self.assertEqual(config.component_type, ComponentType.HAZARD)
        self.assertEqual(config.weight, 0.3)
        self.assertTrue(config.enabled)
        self.assertEqual(config.parameters['param1'], 'value1')
    
    def test_component_config_defaults(self):
        """Test default values in component configuration"""
        config = ComponentConfig(
            name='test_component',
            component_type=ComponentType.EXPOSURE,
            weight=0.2
        )
        
        self.assertTrue(config.enabled)  # Default should be True
        self.assertEqual(config.parameters, {})  # Default should be empty dict


class TestBaseRiskComponent(unittest.TestCase):
    """Test BaseRiskComponent abstract class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ComponentConfig(
            name='test_component',
            component_type=ComponentType.VULNERABILITY,
            weight=0.25,
            parameters={'threshold': 0.5}
        )
    
    def test_base_component_initialization(self):
        """Test base component initialization"""
        # Create a concrete implementation for testing
        class TestComponent(BaseRiskComponent):
            def calculate_score(self, data):
                return 0.5
            
            def get_required_columns(self):
                return ['test_column']
        
        component = TestComponent(self.config)
        
        self.assertEqual(component.name, 'test_component')
        self.assertEqual(component.component_type, ComponentType.VULNERABILITY)
        self.assertEqual(component.weight, 0.25)
        self.assertEqual(component.parameters['threshold'], 0.5)
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns"""
        class TestComponent(BaseRiskComponent):
            def calculate_score(self, data):
                return 0.5
            
            def get_required_columns(self):
                return ['required_column']
        
        component = TestComponent(self.config)
        
        # Test with missing column
        data = pd.DataFrame({'other_column': [1, 2, 3]})
        self.assertFalse(component.validate_data(data))
        
        # Test with required column
        data = pd.DataFrame({'required_column': [1, 2, 3]})
        self.assertTrue(component.validate_data(data))
    
    def test_validate_score_range(self):
        """Test score range validation"""
        class TestComponent(BaseRiskComponent):
            def calculate_score(self, data):
                return 0.5
            
            def get_required_columns(self):
                return []
        
        component = TestComponent(self.config)
        
        # Test valid scores
        self.assertEqual(component._validate_score_range(0.5), 0.5)
        self.assertEqual(component._validate_score_range(0.0), 0.0)
        self.assertEqual(component._validate_score_range(1.0), 1.0)
        
        # Test clipping
        self.assertEqual(component._validate_score_range(-0.1), 0.0)
        self.assertEqual(component._validate_score_range(1.1), 1.0)
        
        # Test with arrays
        scores = np.array([-0.1, 0.5, 1.1])
        clipped = component._validate_score_range(scores)
        np.testing.assert_array_equal(clipped, [0.0, 0.5, 1.0])


class TestResourceTransitionComponent(unittest.TestCase):
    """Test ResourceTransitionComponent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ComponentConfig(
            name='resource_transition',
            component_type=ComponentType.VULNERABILITY,
            weight=0.2
        )
        self.component = ResourceTransitionComponent(self.config)
    
    def test_calculate_score_with_valid_data(self):
        """Test score calculation with valid data"""
        data = pd.DataFrame({
            'renewable_transition_rate': [0.05, 0.1, 0.15],
            'grid_modernization_score': [0.7, 0.5, 0.3]
        })
        
        scores = self.component.calculate_score(data)
        
        # Check that scores are in valid range
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
        
        # Check that higher transition rates give higher scores
        self.assertGreater(scores.iloc[2], scores.iloc[0])
    
    def test_calculate_score_missing_data(self):
        """Test score calculation with missing data"""
        data = pd.DataFrame({'other_column': [1, 2, 3]})
        
        scores = self.component.calculate_score(data)
        
        # Should return baseline scores
        self.assertEqual(len(scores), 3)
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_required_columns(self):
        """Test required columns specification"""
        required = self.component.get_required_columns()
        self.assertIn('renewable_transition_rate', required)


class TestLoadGrowthComponent(unittest.TestCase):
    """Test LoadGrowthComponent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ComponentConfig(
            name='load_growth',
            component_type=ComponentType.EXPOSURE,
            weight=0.3
        )
        self.component = LoadGrowthComponent(self.config)
    
    def test_calculate_score_with_full_data(self):
        """Test score calculation with complete data"""
        data = pd.DataFrame({
            'projected_load_growth_rate': [0.02, 0.05, 0.08],
            'economic_growth_rate': [0.01, 0.03, 0.05],
            'population_growth_rate': [0.005, 0.015, 0.025],
            'industrial_growth_rate': [0.02, 0.04, 0.06]
        })
        
        scores = self.component.calculate_score(data)
        
        # Check that scores are in valid range
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
        
        # Check that higher growth rates give higher scores
        self.assertGreater(scores.iloc[2], scores.iloc[0])
    
    def test_calculate_score_minimal_data(self):
        """Test score calculation with minimal required data"""
        data = pd.DataFrame({
            'projected_load_growth_rate': [0.03, 0.06]
        })
        
        scores = self.component.calculate_score(data)
        
        # Should still work with just required column
        self.assertEqual(len(scores), 2)
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())


class TestExtensibilityManager(unittest.TestCase):
    """Test ExtensibilityManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ExtensibilityManager()
        
        # Create test components
        self.hazard_config = ComponentConfig(
            name='test_hazard',
            component_type=ComponentType.HAZARD,
            weight=0.2
        )
        
        self.exposure_config = ComponentConfig(
            name='test_exposure',
            component_type=ComponentType.EXPOSURE,
            weight=0.3
        )
        
        # Mock components for testing
        self.hazard_component = Mock(spec=BaseRiskComponent)
        self.hazard_component.get_component_name.return_value = 'test_hazard'
        self.hazard_component.get_component_type.return_value = ComponentType.HAZARD
        self.hazard_component.config = self.hazard_config
        self.hazard_component.enabled = True
        self.hazard_component.weight = 0.2
        
        self.exposure_component = Mock(spec=BaseRiskComponent)
        self.exposure_component.get_component_name.return_value = 'test_exposure'
        self.exposure_component.get_component_type.return_value = ComponentType.EXPOSURE
        self.exposure_component.config = self.exposure_config
        self.exposure_component.enabled = True
        self.exposure_component.weight = 0.3
    
    def test_register_component(self):
        """Test component registration"""
        self.manager.register_component(self.hazard_component)
        
        self.assertIn('test_hazard', self.manager.components)
        self.assertEqual(
            self.manager.get_component('test_hazard'),
            self.hazard_component
        )
    
    def test_unregister_component(self):
        """Test component unregistration"""
        self.manager.register_component(self.hazard_component)
        self.manager.unregister_component('test_hazard')
        
        self.assertNotIn('test_hazard', self.manager.components)
        self.assertIsNone(self.manager.get_component('test_hazard'))
    
    def test_list_components_by_type(self):
        """Test listing components by type"""
        self.manager.register_component(self.hazard_component)
        self.manager.register_component(self.exposure_component)
        
        hazard_components = self.manager.list_components(ComponentType.HAZARD)
        exposure_components = self.manager.list_components(ComponentType.EXPOSURE)
        all_components = self.manager.list_components()
        
        self.assertIn('test_hazard', hazard_components)
        self.assertNotIn('test_exposure', hazard_components)
        
        self.assertIn('test_exposure', exposure_components)
        self.assertNotIn('test_hazard', exposure_components)
        
        self.assertIn('test_hazard', all_components)
        self.assertIn('test_exposure', all_components)
    
    def test_calculate_component_scores(self):
        """Test calculating scores for registered components"""
        # Set up mock return values
        self.hazard_component.calculate_score.return_value = pd.Series([0.6, 0.7])
        self.exposure_component.calculate_score.return_value = pd.Series([0.4, 0.5])
        
        self.manager.register_component(self.hazard_component)
        self.manager.register_component(self.exposure_component)
        
        test_data = pd.DataFrame({'test_col': [1, 2]})
        
        # Test calculating all scores
        all_scores = self.manager.calculate_component_scores(test_data)
        self.assertIn('test_hazard', all_scores)
        self.assertIn('test_exposure', all_scores)
        
        # Test calculating hazard scores only
        hazard_scores = self.manager.calculate_component_scores(
            test_data, ComponentType.HAZARD
        )
        self.assertIn('test_hazard', hazard_scores)
        self.assertNotIn('test_exposure', hazard_scores)
    
    def test_load_components_from_config(self):
        """Test loading components from configuration"""
        config = {
            'extended_components': {
                'resource_transition': {
                    'type': 'vulnerability',
                    'weight': 0.15,
                    'enabled': True,
                    'parameters': {'threshold': 0.1}
                },
                'load_growth': {
                    'type': 'exposure',
                    'weight': 0.25,
                    'enabled': True,
                    'parameters': {'high_growth_threshold': 0.04}
                }
            }
        }
        
        self.manager.load_components_from_config(config)
        
        # Check that components were loaded
        self.assertIsNotNone(self.manager.get_component('resource_transition'))
        self.assertIsNotNone(self.manager.get_component('load_growth'))


class TestExtendedRiskScoringEngine(unittest.TestCase):
    """Test ExtendedRiskScoringEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'weights': {
                'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15,
                'pop': 0.7, 'load': 0.3,
                'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1,
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            },
            'extended_weights': {
                'core_hazard_weight': 0.8, 'extended_hazard_weight': 0.2,
                'core_exposure_weight': 0.8, 'extended_exposure_weight': 0.2,
                'core_vulnerability_weight': 0.8, 'extended_vulnerability_weight': 0.2
            },
            'extended_components': {
                'resource_transition': {
                    'type': 'vulnerability',
                    'weight': 0.5,
                    'enabled': True,
                    'parameters': {}
                }
            }
        }
        
        self.engine = ExtendedRiskScoringEngine(self.config)
    
    def test_initialization(self):
        """Test extended engine initialization"""
        self.assertIsNotNone(self.engine.core_engine)
        self.assertIsNotNone(self.engine.extensibility_manager)
        self.assertEqual(self.engine.extended_weights.core_hazard_weight, 0.8)
    
    def test_calculate_extended_hazard_score(self):
        """Test extended hazard score calculation"""
        weather_data = pd.DataFrame({
            'thermal_stress': [0.3, 0.6, 0.9],
            'wind_stress': [0.2, 0.5, 0.8],
            'precip_stress': [0.1, 0.3, 0.7],
            'storm_proxy': [0.0, 0.2, 0.6]
        })
        
        result = self.engine.calculate_extended_hazard_score(weather_data)
        
        self.assertIn('extended_hazard_score', result.columns)
        self.assertTrue((result['extended_hazard_score'] >= 0).all())
        self.assertTrue((result['extended_hazard_score'] <= 1).all())
    
    def test_validate_extended_system(self):
        """Test extended system validation"""
        weather_data = pd.DataFrame({
            'thermal_stress': [0.3, 0.6],
            'wind_stress': [0.2, 0.5],
            'precip_stress': [0.1, 0.3],
            'storm_proxy': [0.0, 0.2]
        })
        
        infrastructure_data = pd.DataFrame({
            'normalized_pop_density': [0.4, 0.7],
            'renewable_share': [0.3, 0.6],
            'transmission_scarcity': [0.5, 0.5],
            'outage_flag': [False, False]
        })
        
        validation = self.engine.validate_extended_system(
            weather_data, infrastructure_data
        )
        
        self.assertIn('core_system_valid', validation)
        self.assertIn('extended_components_valid', validation)
        self.assertIn('integration_valid', validation)


class TestExtensionExamples(unittest.TestCase):
    """Test extension example components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = create_sample_extension_data()
    
    def test_cyber_security_component(self):
        """Test CyberSecurityRiskComponent"""
        config = ComponentConfig(
            name='cyber_security',
            component_type=ComponentType.VULNERABILITY,
            weight=0.2
        )
        component = CyberSecurityRiskComponent(config)
        
        scores = component.calculate_score(self.sample_data)
        
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_market_volatility_component(self):
        """Test MarketVolatilityComponent"""
        config = ComponentConfig(
            name='market_volatility',
            component_type=ComponentType.EXPOSURE,
            weight=0.25
        )
        component = MarketVolatilityComponent(config)
        
        scores = component.calculate_score(self.sample_data)
        
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_seasonal_demand_component(self):
        """Test SeasonalDemandComponent"""
        config = ComponentConfig(
            name='seasonal_demand',
            component_type=ComponentType.HAZARD,
            weight=0.15
        )
        component = SeasonalDemandComponent(config)
        
        scores = component.calculate_score(self.sample_data)
        
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_infrastructure_age_component(self):
        """Test InfrastructureAgeComponent"""
        config = ComponentConfig(
            name='infrastructure_age',
            component_type=ComponentType.VULNERABILITY,
            weight=0.3
        )
        component = InfrastructureAgeComponent(config)
        
        scores = component.calculate_score(self.sample_data)
        
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""
    
    def test_full_extended_pipeline(self):
        """Test complete extended risk scoring pipeline"""
        # Create extended engine with multiple components
        config = {
            'weights': {
                'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15,
                'pop': 0.7, 'load': 0.3,
                'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1,
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            },
            'extended_components': {
                'resource_transition': {
                    'type': 'vulnerability',
                    'weight': 0.4,
                    'enabled': True,
                    'parameters': {}
                },
                'load_growth': {
                    'type': 'exposure',
                    'weight': 0.6,
                    'enabled': True,
                    'parameters': {}
                }
            }
        }
        
        engine = ExtendedRiskScoringEngine(config)
        
        # Create test data
        weather_data = pd.DataFrame({
            'thermal_stress': [0.3, 0.6, 0.9],
            'wind_stress': [0.2, 0.5, 0.8],
            'precip_stress': [0.1, 0.3, 0.7],
            'storm_proxy': [0.0, 0.2, 0.6]
        })
        
        infrastructure_data = pd.DataFrame({
            'normalized_pop_density': [0.4, 0.7, 0.9],
            'renewable_share': [0.3, 0.6, 0.8],
            'transmission_scarcity': [0.5, 0.5, 0.5],
            'outage_flag': [False, False, True],
            'renewable_transition_rate': [0.05, 0.1, 0.15],
            'projected_load_growth_rate': [0.02, 0.05, 0.08]
        })
        
        # Calculate extended scores
        hazard_result = engine.calculate_extended_hazard_score(weather_data)
        exposure_result = engine.calculate_extended_exposure_score(infrastructure_data)
        vulnerability_result = engine.calculate_extended_vulnerability_score(infrastructure_data)
        
        # Verify results
        self.assertIn('extended_hazard_score', hazard_result.columns)
        self.assertIn('extended_exposure_score', exposure_result.columns)
        self.assertIn('extended_vulnerability_score', vulnerability_result.columns)
        
        # Check score ranges
        self.assertTrue((hazard_result['extended_hazard_score'] >= 0).all())
        self.assertTrue((hazard_result['extended_hazard_score'] <= 1).all())
        self.assertTrue((exposure_result['extended_exposure_score'] >= 0).all())
        self.assertTrue((exposure_result['extended_exposure_score'] <= 1).all())
        self.assertTrue((vulnerability_result['extended_vulnerability_score'] >= 0).all())
        self.assertTrue((vulnerability_result['extended_vulnerability_score'] <= 1).all())


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)