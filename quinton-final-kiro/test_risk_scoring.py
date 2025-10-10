"""
Unit tests for risk scoring engine.

Tests validate hazard, exposure, vulnerability, and final risk calculations
according to the requirements, including weight sensitivity and mathematical correctness.
"""

import unittest
import numpy as np
import pandas as pd
from risk_scoring_engine import (
    RiskScoringEngine,
    RiskWeights,
    RiskAssessment,
    test_weight_sensitivity
)


class TestRiskWeights(unittest.TestCase):
    """Test risk weights configuration and validation"""
    
    def test_default_weights(self):
        """Test default weight configuration"""
        weights = RiskWeights()
        
        # Check hazard weights sum to 1.0
        hazard_sum = weights.thermal + weights.wind + weights.precip + weights.storm
        self.assertAlmostEqual(hazard_sum, 1.0, places=3)
        
        # Check exposure weights sum to 1.0
        exposure_sum = weights.pop + weights.load
        self.assertAlmostEqual(exposure_sum, 1.0, places=3)
        
        # Check vulnerability weights sum to 1.0
        vulnerability_sum = weights.renew_share + weights.tx_scarcity + weights.outage
        self.assertAlmostEqual(vulnerability_sum, 1.0, places=3)
        
        # Check blend weights sum to 1.0
        blend_sum = weights.alpha + weights.beta + weights.gamma
        self.assertAlmostEqual(blend_sum, 1.0, places=3)
    
    def test_custom_weights(self):
        """Test custom weight configuration"""
        custom_weights = RiskWeights(
            thermal=0.4, wind=0.3, precip=0.2, storm=0.1,
            pop=0.8, load=0.2,
            renew_share=0.5, tx_scarcity=0.4, outage=0.1,
            alpha=0.6, beta=0.25, gamma=0.15
        )
        
        self.assertEqual(custom_weights.thermal, 0.4)
        self.assertEqual(custom_weights.alpha, 0.6)


class TestHazardScoreCalculation(unittest.TestCase):
    """Test hazard score calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
    
    def test_hazard_score_calculation(self):
        """Test basic hazard score calculation"""
        # Test with equal stress values
        hazard_score = self.engine.calculate_hazard_score(0.5, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(hazard_score, 0.5, places=3)
        
        # Test with maximum stress values
        hazard_score_max = self.engine.calculate_hazard_score(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(hazard_score_max, 1.0, places=3)
        
        # Test with minimum stress values
        hazard_score_min = self.engine.calculate_hazard_score(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(hazard_score_min, 0.0, places=3)
    
    def test_hazard_score_weights(self):
        """Test that hazard score respects configured weights"""
        # Test thermal dominance
        thermal_dominant = self.engine.calculate_hazard_score(1.0, 0.0, 0.0, 0.0)
        expected_thermal = self.engine.weights.thermal * 1.0
        self.assertAlmostEqual(thermal_dominant, expected_thermal, places=3)
        
        # Test wind dominance
        wind_dominant = self.engine.calculate_hazard_score(0.0, 1.0, 0.0, 0.0)
        expected_wind = self.engine.weights.wind * 1.0
        self.assertAlmostEqual(wind_dominant, expected_wind, places=3)
    
    def test_hazard_score_array_input(self):
        """Test hazard score calculation with numpy arrays"""
        thermal_stress = np.array([0.0, 0.5, 1.0])
        wind_stress = np.array([0.2, 0.6, 0.8])
        precip_stress = np.array([0.1, 0.3, 0.9])
        storm_proxy = np.array([0.0, 0.4, 1.0])
        
        results = self.engine.calculate_hazard_score(
            thermal_stress, wind_stress, precip_stress, storm_proxy
        )
        
        # Check that results are numpy array
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 3)
        
        # Check that all values are in [0,1] range
        self.assertTrue((results >= 0.0).all())
        self.assertTrue((results <= 1.0).all())
    
    def test_hazard_score_pandas_input(self):
        """Test hazard score calculation with pandas Series"""
        df = pd.DataFrame({
            'thermal': [0.0, 0.5, 1.0],
            'wind': [0.2, 0.6, 0.8],
            'precip': [0.1, 0.3, 0.9],
            'storm': [0.0, 0.4, 1.0]
        })
        
        results = self.engine.calculate_hazard_score(
            df['thermal'], df['wind'], df['precip'], df['storm']
        )
        
        # Check that results are pandas Series
        self.assertIsInstance(results, pd.Series)
        self.assertEqual(len(results), 3)
    
    def test_process_hazard_scores(self):
        """Test complete hazard score processing"""
        weather_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'thermal_stress': [0.2, 0.6, 0.9],
            'wind_stress': [0.1, 0.5, 0.8],
            'precip_stress': [0.0, 0.3, 0.7],
            'storm_proxy': [0.0, 0.2, 0.6]
        })
        
        result = self.engine.process_hazard_scores(weather_data)
        
        # Check that hazard_score column is added
        self.assertIn('hazard_score', result.columns)
        
        # Check that all original columns are preserved
        for col in weather_data.columns:
            self.assertIn(col, result.columns)
        
        # Validate hazard scores
        validation = self.engine.validate_hazard_calculation(result)
        self.assertTrue(validation['hazard_range_valid'])
        self.assertTrue(validation['hazard_no_nan'])
    
    def test_get_top_hazard_contributors(self):
        """Test identification of top hazard contributors"""
        # Test case where thermal is dominant
        contributors = self.engine.get_top_hazard_contributors(1.0, 0.2, 0.1, 0.0)
        self.assertEqual(contributors[0], 'thermal')
        
        # Test case where wind is dominant
        contributors = self.engine.get_top_hazard_contributors(0.1, 1.0, 0.2, 0.0)
        self.assertEqual(contributors[0], 'wind')


class TestExposureScoreCalculation(unittest.TestCase):
    """Test exposure score calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
    
    def test_exposure_score_with_load_factor(self):
        """Test exposure score calculation with load factor"""
        exposure_score = self.engine.calculate_exposure_score(0.6, 0.8)
        expected = self.engine.weights.pop * 0.6 + self.engine.weights.load * 0.8
        self.assertAlmostEqual(exposure_score, expected, places=3)
    
    def test_exposure_score_without_load_factor(self):
        """Test exposure score calculation without load factor (population-only)"""
        exposure_score = self.engine.calculate_exposure_score(0.7, None)
        self.assertAlmostEqual(exposure_score, 0.7, places=3)
    
    def test_exposure_score_array_input(self):
        """Test exposure score calculation with arrays"""
        pop_density = np.array([0.2, 0.6, 0.9])
        load_factor = np.array([0.1, 0.5, 0.8])
        
        results = self.engine.calculate_exposure_score(pop_density, load_factor)
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 3)
        self.assertTrue((results >= 0.0).all())
        self.assertTrue((results <= 1.0).all())
    
    def test_process_exposure_scores(self):
        """Test complete exposure score processing"""
        infrastructure_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'normalized_pop_density': [0.2, 0.6, 0.9],
            'load_factor': [0.1, 0.5, 0.8]
        })
        
        result = self.engine.process_exposure_scores(infrastructure_data)
        
        # Check that exposure_score column is added
        self.assertIn('exposure_score', result.columns)
        
        # Validate exposure scores
        validation = self.engine.validate_exposure_calculation(result)
        self.assertTrue(validation['exposure_range_valid'])
        self.assertTrue(validation['exposure_no_nan'])
    
    def test_validate_exposure_urban_rural(self):
        """Test exposure validation against urban vs rural patterns"""
        data = pd.DataFrame({
            'cell_id': ['urban1', 'urban2', 'rural1', 'rural2'],
            'normalized_pop_density': [0.9, 0.8, 0.1, 0.2],
            'exposure_score': [0.85, 0.75, 0.15, 0.25]
        })
        
        validation = self.engine.validate_exposure_against_urban_rural(data)
        
        self.assertTrue(validation['data_available'])
        self.assertTrue(validation['urban_higher_than_rural'])
        self.assertTrue(validation['positive_correlation'])


class TestVulnerabilityScoreCalculation(unittest.TestCase):
    """Test vulnerability score calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
    
    def test_vulnerability_score_calculation(self):
        """Test basic vulnerability score calculation"""
        vuln_score = self.engine.calculate_vulnerability_score(0.6, 0.4, False)
        expected = (self.engine.weights.renew_share * 0.6 + 
                   self.engine.weights.tx_scarcity * 0.4 + 
                   self.engine.weights.outage * 0.0)
        self.assertAlmostEqual(vuln_score, expected, places=3)
    
    def test_vulnerability_score_with_outage(self):
        """Test vulnerability score with outage flag"""
        vuln_score = self.engine.calculate_vulnerability_score(0.5, 0.5, True)
        expected = (self.engine.weights.renew_share * 0.5 + 
                   self.engine.weights.tx_scarcity * 0.5 + 
                   self.engine.weights.outage * 1.0)
        self.assertAlmostEqual(vuln_score, expected, places=3)
    
    def test_vulnerability_score_array_input(self):
        """Test vulnerability score calculation with arrays"""
        renewable_share = np.array([0.2, 0.6, 0.9])
        tx_scarcity = np.array([0.3, 0.5, 0.7])
        outage_flag = np.array([False, True, False])
        
        results = self.engine.calculate_vulnerability_score(
            renewable_share, tx_scarcity, outage_flag
        )
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 3)
        self.assertTrue((results >= 0.0).all())
        self.assertTrue((results <= 1.0).all())
    
    def test_process_vulnerability_scores(self):
        """Test complete vulnerability score processing"""
        infrastructure_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'renewable_share': [0.2, 0.6, 0.9],
            'transmission_scarcity': [0.3, 0.5, 0.7],
            'outage_flag': [False, True, False]
        })
        
        result = self.engine.process_vulnerability_scores(infrastructure_data)
        
        # Check that vulnerability_score column is added
        self.assertIn('vulnerability_score', result.columns)
        
        # Validate vulnerability scores
        validation = self.engine.validate_vulnerability_calculation(result)
        self.assertTrue(validation['vulnerability_range_valid'])
        self.assertTrue(validation['vulnerability_no_nan'])
    
    def test_process_vulnerability_scores_missing_data(self):
        """Test vulnerability processing with missing transmission/outage data"""
        infrastructure_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'renewable_share': [0.2, 0.6, 0.9]
            # Missing transmission_scarcity and outage_flag
        })
        
        result = self.engine.process_vulnerability_scores(infrastructure_data)
        
        # Check that baseline values are used
        self.assertIn('transmission_scarcity', result.columns)
        self.assertIn('outage_flag', result.columns)
        self.assertTrue((result['transmission_scarcity'] == 0.5).all())
        self.assertTrue((result['outage_flag'] == False).all())
    
    def test_validate_vulnerability_edge_cases(self):
        """Test vulnerability edge case validation"""
        data = pd.DataFrame({
            'cell_id': ['low_ren', 'high_ren', 'low_tx', 'high_tx'],
            'renewable_share': [0.1, 0.9, 0.5, 0.5],
            'transmission_scarcity': [0.5, 0.5, 0.1, 0.9],
            'vulnerability_score': [0.3, 0.7, 0.25, 0.65]
        })
        
        validation = self.engine.validate_vulnerability_edge_cases(data)
        
        self.assertTrue(validation['data_available'])
        # High renewable share should increase vulnerability
        self.assertTrue(validation['high_renewable_more_vulnerable'])
        # High transmission scarcity should increase vulnerability
        self.assertTrue(validation['high_tx_scarcity_more_vulnerable'])


class TestFinalRiskScoreCalculation(unittest.TestCase):
    """Test final risk score calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
    
    def test_final_risk_score_single_value(self):
        """Test final risk score calculation for single value"""
        # Single value cannot be z-scored, should return combined score
        risk_score = self.engine.calculate_final_risk_score(0.6, 0.4, 0.5)
        expected = (self.engine.weights.alpha * 0.6 + 
                   self.engine.weights.beta * 0.4 + 
                   self.engine.weights.gamma * 0.5)
        self.assertAlmostEqual(risk_score, expected, places=3)
    
    def test_final_risk_score_array_zscore(self):
        """Test final risk score z-score calculation with arrays"""
        hazard_scores = np.array([0.2, 0.5, 0.8])
        exposure_scores = np.array([0.3, 0.6, 0.9])
        vulnerability_scores = np.array([0.1, 0.4, 0.7])
        
        risk_scores = self.engine.calculate_final_risk_score(
            hazard_scores, exposure_scores, vulnerability_scores
        )
        
        # Check that z-scores have mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(np.mean(risk_scores), 0.0, places=10)
        self.assertAlmostEqual(np.std(risk_scores, ddof=1), 1.0, places=10)
    
    def test_calculate_final_risk_scores_by_horizon(self):
        """Test final risk score calculation by forecast horizon"""
        combined_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C', 'A', 'B', 'C'],
            'horizon_h': [12, 12, 12, 24, 24, 24],
            'hazard_score': [0.2, 0.5, 0.8, 0.3, 0.6, 0.9],
            'exposure_score': [0.3, 0.6, 0.9, 0.4, 0.7, 1.0],
            'vulnerability_score': [0.1, 0.4, 0.7, 0.2, 0.5, 0.8]
        })
        
        result = self.engine.calculate_final_risk_scores_by_horizon(combined_data)
        
        # Check that final_risk column is added
        self.assertIn('final_risk', result.columns)
        
        # Check z-score properties for each horizon
        for horizon in [12, 24]:
            horizon_data = result[result['horizon_h'] == horizon]['final_risk']
            self.assertAlmostEqual(horizon_data.mean(), 0.0, places=10)
            self.assertAlmostEqual(horizon_data.std(ddof=1), 1.0, places=10)
    
    def test_get_top_risk_contributors(self):
        """Test identification of top risk contributors"""
        # Test case where hazard is dominant
        contributors = self.engine.get_top_risk_contributors(0.9, 0.3, 0.2)
        self.assertEqual(contributors[0], 'hazard')
        
        # Test case where exposure is dominant
        contributors = self.engine.get_top_risk_contributors(0.2, 0.9, 0.1)
        self.assertEqual(contributors[0], 'exposure')
    
    def test_validate_risk_score_distribution(self):
        """Test risk score distribution validation"""
        data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C', 'A', 'B', 'C'],
            'horizon_h': [12, 12, 12, 24, 24, 24],
            'hazard_score': [0.2, 0.5, 0.8, 0.3, 0.6, 0.9],
            'exposure_score': [0.3, 0.6, 0.9, 0.4, 0.7, 1.0],
            'vulnerability_score': [0.1, 0.4, 0.7, 0.2, 0.5, 0.8],
            'final_risk': [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]  # Mock z-scores
        })
        
        validation = self.engine.validate_risk_score_distribution(data)
        
        self.assertTrue(validation['final_risk_exists'])
        self.assertTrue(validation['risk_no_nan'])
    
    def test_create_complete_risk_assessment(self):
        """Test complete risk assessment creation"""
        weather_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'horizon_h': [12, 12, 12],
            'thermal_stress': [0.2, 0.6, 0.9],
            'wind_stress': [0.1, 0.5, 0.8],
            'precip_stress': [0.0, 0.3, 0.7],
            'storm_proxy': [0.0, 0.2, 0.6]
        })
        
        infrastructure_data = pd.DataFrame({
            'cell_id': ['A', 'B', 'C'],
            'normalized_pop_density': [0.2, 0.6, 0.9],
            'renewable_share': [0.3, 0.7, 0.9],
            'transmission_scarcity': [0.4, 0.5, 0.6]
        })
        
        result = self.engine.create_complete_risk_assessment(
            weather_data, infrastructure_data
        )
        
        # Check that all score columns are present
        expected_columns = [
            'hazard_score', 'exposure_score', 'vulnerability_score', 'final_risk'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that all cells are included
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result['cell_id']), {'A', 'B', 'C'})


class TestWeightSensitivity(unittest.TestCase):
    """Test weight sensitivity analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
        self.base_scores = {
            'thermal': 0.6,
            'wind': 0.4,
            'precip': 0.3,
            'storm': 0.2
        }
    
    def test_weight_sensitivity_analysis(self):
        """Test weight sensitivity analysis function"""
        weight_variations = {
            'thermal': [0.1, 0.5, 0.7],
            'wind': [0.1, 0.4, 0.6]
        }
        
        results = test_weight_sensitivity(
            self.engine, self.base_scores, weight_variations
        )
        
        # Check that results DataFrame has expected structure
        expected_columns = [
            'scenario', 'thermal_weight', 'wind_weight', 'precip_weight', 
            'storm_weight', 'hazard_score', 'score_change'
        ]
        for col in expected_columns:
            self.assertIn(col, results.columns)
        
        # Check that base case has zero score change
        base_row = results[results['scenario'] == 'base']
        self.assertEqual(len(base_row), 1)
        self.assertAlmostEqual(base_row['score_change'].iloc[0], 0.0, places=6)
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized in sensitivity analysis"""
        weight_variations = {'thermal': [0.5]}
        
        results = test_weight_sensitivity(
            self.engine, self.base_scores, weight_variations
        )
        
        # Check that hazard weights still sum to 1.0 after modification
        modified_row = results[results['scenario'] == 'thermal_0.5']
        weight_sum = (modified_row['thermal_weight'].iloc[0] + 
                     modified_row['wind_weight'].iloc[0] + 
                     modified_row['precip_weight'].iloc[0] + 
                     modified_row['storm_weight'].iloc[0])
        
        self.assertAlmostEqual(weight_sum, 1.0, places=3)


class TestRiskScoringEngineIntegration(unittest.TestCase):
    """Test complete risk scoring engine integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RiskScoringEngine()
    
    def test_engine_initialization(self):
        """Test risk scoring engine initialization"""
        # Test with default config
        engine_default = RiskScoringEngine()
        self.assertIsNotNone(engine_default.weights)
        
        # Test with custom config
        custom_config = {
            'weights': {
                'thermal': 0.4, 'wind': 0.3, 'precip': 0.2, 'storm': 0.1,
                'pop': 0.8, 'load': 0.2,
                'renew_share': 0.5, 'tx_scarcity': 0.4, 'outage': 0.1,
                'alpha': 0.6, 'beta': 0.25, 'gamma': 0.15
            }
        }
        engine_custom = RiskScoringEngine(custom_config)
        self.assertEqual(engine_custom.weights.thermal, 0.4)
    
    def test_weight_validation(self):
        """Test weight validation during initialization"""
        # Test with invalid weights (negative)
        invalid_config = {
            'weights': {
                'thermal': -0.1, 'wind': 0.3, 'precip': 0.2, 'storm': 0.1,
                'pop': 0.7, 'load': 0.3,
                'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1,
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            }
        }
        
        with self.assertRaises(ValueError):
            RiskScoringEngine(invalid_config)
    
    def test_summary_statistics(self):
        """Test summary statistics functions"""
        # Create sample data
        data = pd.DataFrame({
            'hazard_score': [0.2, 0.5, 0.8],
            'exposure_score': [0.3, 0.6, 0.9],
            'vulnerability_score': [0.1, 0.4, 0.7],
            'final_risk': [-1.0, 0.0, 1.0]
        })
        
        # Test hazard summary
        hazard_summary = self.engine.get_hazard_summary_statistics(data)
        self.assertIn('mean', hazard_summary)
        self.assertIn('std', hazard_summary)
        
        # Test exposure summary
        exposure_summary = self.engine.get_exposure_summary_statistics(data)
        self.assertIn('mean', exposure_summary)
        
        # Test vulnerability summary
        vulnerability_summary = self.engine.get_vulnerability_summary_statistics(data)
        self.assertIn('mean', vulnerability_summary)
        
        # Test final risk summary
        risk_summary = self.engine.get_final_risk_summary_statistics(data)
        self.assertIn('mean', risk_summary)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)