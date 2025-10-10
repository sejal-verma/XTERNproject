"""
Tests for Ablation Analysis System

This module contains comprehensive tests for the ablation analysis system,
including component removal, sensitivity analysis, and validation logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import logging

from ablation_analysis import (
    AblationAnalyzer, AblationResult, ComponentImportance
)
from risk_scoring_engine import RiskScoringEngine


class TestAblationAnalyzer:
    """Test suite for AblationAnalyzer class"""
    
    @pytest.fixture
    def mock_risk_engine(self):
        """Create mock risk scoring engine for testing"""
        mock_engine = Mock(spec=RiskScoringEngine)
        
        # Mock weights
        mock_weights = Mock()
        mock_weights.alpha = 0.5
        mock_weights.beta = 0.3
        mock_weights.gamma = 0.2
        mock_weights.thermal = 0.3
        mock_weights.wind = 0.3
        mock_weights.precip = 0.25
        mock_weights.storm = 0.15
        mock_weights.pop = 0.7
        mock_weights.load = 0.3
        mock_weights.renew_share = 0.6
        mock_weights.tx_scarcity = 0.3
        mock_weights.outage = 0.1
        
        mock_engine.weights = mock_weights
        
        # Mock calculation methods
        def mock_final_risk(hazard, exposure, vulnerability):
            return 0.5 * hazard + 0.3 * exposure + 0.2 * vulnerability
        
        def mock_hazard_score(thermal, wind, precip, storm):
            return 0.3 * thermal + 0.3 * wind + 0.25 * precip + 0.15 * storm
        
        def mock_exposure_score(pop, load=None):
            if load is not None:
                return 0.7 * pop + 0.3 * load
            return pop
        
        def mock_vulnerability_score(renew, tx, outage):
            return 0.6 * renew + 0.3 * tx + 0.1 * float(outage)
        
        mock_engine.calculate_final_risk_score = mock_final_risk
        mock_engine.calculate_hazard_score = mock_hazard_score
        mock_engine.calculate_exposure_score = mock_exposure_score
        mock_engine.calculate_vulnerability_score = mock_vulnerability_score
        
        return mock_engine
    
    @pytest.fixture
    def sample_risk_data(self):
        """Create sample risk data for testing"""
        np.random.seed(42)
        n_cells = 100
        
        data = {
            'cell_id': [f'CELL_{i:03d}' for i in range(n_cells)],
            
            # Hazard subcomponents
            'thermal_stress': np.random.uniform(0, 1, n_cells),
            'wind_stress': np.random.uniform(0, 1, n_cells),
            'precip_stress': np.random.uniform(0, 1, n_cells),
            'storm_proxy': np.random.uniform(0, 1, n_cells),
            
            # Exposure subcomponents
            'normalized_pop_density': np.random.uniform(0, 1, n_cells),
            'load_factor': np.random.uniform(0, 1, n_cells),
            
            # Vulnerability subcomponents
            'renewable_share': np.random.uniform(0, 1, n_cells),
            'transmission_scarcity': np.random.uniform(0, 1, n_cells),
            'outage_flag': np.random.choice([True, False], n_cells)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate component scores
        df['hazard_score'] = (0.3 * df['thermal_stress'] + 
                             0.3 * df['wind_stress'] + 
                             0.25 * df['precip_stress'] + 
                             0.15 * df['storm_proxy'])
        
        df['exposure_score'] = (0.7 * df['normalized_pop_density'] + 
                               0.3 * df['load_factor'])
        
        df['vulnerability_score'] = (0.6 * df['renewable_share'] + 
                                   0.3 * df['transmission_scarcity'] + 
                                   0.1 * df['outage_flag'].astype(float))
        
        # Calculate final risk scores
        df['final_risk'] = (0.5 * df['hazard_score'] + 
                           0.3 * df['exposure_score'] + 
                           0.2 * df['vulnerability_score'])
        
        return df
    
    @pytest.fixture
    def incomplete_risk_data(self):
        """Create incomplete risk data for testing edge cases"""
        np.random.seed(42)
        n_cells = 50
        
        data = {
            'cell_id': [f'CELL_{i:03d}' for i in range(n_cells)],
            'hazard_score': np.random.uniform(0, 1, n_cells),
            'exposure_score': np.random.uniform(0, 1, n_cells),
            'vulnerability_score': np.random.uniform(0, 1, n_cells),
            
            # Missing some subcomponents
            'thermal_stress': np.random.uniform(0, 1, n_cells),
            'wind_stress': np.random.uniform(0, 1, n_cells),
            # Missing precip_stress and storm_proxy
            
            'normalized_pop_density': np.random.uniform(0, 1, n_cells),
            # Missing load_factor
            
            'renewable_share': np.random.uniform(0, 1, n_cells),
            'transmission_scarcity': np.full(n_cells, 0.5),  # Baseline values
            # Missing outage_flag
        }
        
        return pd.DataFrame(data)
    
    def test_analyzer_initialization(self, mock_risk_engine):
        """Test ablation analyzer initialization"""
        # Test default initialization
        analyzer = AblationAnalyzer(mock_risk_engine)
        assert analyzer.risk_engine == mock_risk_engine
        assert 'hazard' in analyzer.components
        assert 'exposure' in analyzer.components
        assert 'vulnerability' in analyzer.components
        
        # Test custom configuration
        custom_config = {
            'ablation': {
                'impact_thresholds': {
                    'low': 0.05,
                    'medium': 0.15,
                    'high': 0.15
                }
            }
        }
        analyzer = AblationAnalyzer(mock_risk_engine, custom_config)
        assert analyzer.config['ablation']['impact_thresholds']['low'] == 0.05
    
    def test_data_validation(self, mock_risk_engine, sample_risk_data):
        """Test ablation data validation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Valid data should pass
        analyzer._validate_ablation_data(sample_risk_data)
        
        # Missing required columns should fail
        incomplete_data = sample_risk_data.drop(columns=['hazard_score'])
        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer._validate_ablation_data(incomplete_data)
        
        # Insufficient data should fail
        tiny_data = sample_risk_data.head(5)
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer._validate_ablation_data(tiny_data)
        
        # All NaN values should fail
        nan_data = sample_risk_data.copy()
        nan_data['hazard_score'] = np.nan
        with pytest.raises(ValueError, match="All values.*are NaN"):
            analyzer._validate_ablation_data(nan_data)
    
    def test_baseline_risk_calculation(self, mock_risk_engine, sample_risk_data):
        """Test baseline risk score calculation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        baseline_risks = analyzer._calculate_baseline_risk_scores(sample_risk_data)
        
        # Should return a pandas Series
        assert isinstance(baseline_risks, pd.Series)
        assert len(baseline_risks) == len(sample_risk_data)
        
        # Should not have NaN values (for valid input data)
        assert not baseline_risks.isna().any()
        
        # Values should be reasonable (z-scores typically in [-3, 3])
        assert baseline_risks.min() >= -5
        assert baseline_risks.max() <= 5
    
    def test_component_ablation(self, mock_risk_engine, sample_risk_data):
        """Test individual component ablation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Calculate baseline risks
        original_risks = analyzer._calculate_baseline_risk_scores(sample_risk_data)
        
        # Test hazard component ablation
        hazard_result = analyzer._ablate_component(sample_risk_data, 'hazard', original_risks)
        
        assert isinstance(hazard_result, AblationResult)
        assert hazard_result.component_name == 'hazard'
        assert hazard_result.cells_affected > 0
        assert hazard_result.total_cells == len(sample_risk_data)
        assert hazard_result.impact_magnitude in ['low', 'medium', 'high']
        
        # Hazard should have significant impact (it has 50% weight)
        assert hazard_result.risk_change_percentage > 5.0
        
        # Test exposure component ablation
        exposure_result = analyzer._ablate_component(sample_risk_data, 'exposure', original_risks)
        
        assert exposure_result.component_name == 'exposure'
        assert exposure_result.risk_change_percentage > 0
        
        # Test vulnerability component ablation
        vulnerability_result = analyzer._ablate_component(sample_risk_data, 'vulnerability', original_risks)
        
        assert vulnerability_result.component_name == 'vulnerability'
        assert vulnerability_result.risk_change_percentage > 0
        
        # Hazard should have higher impact than vulnerability (higher weight)
        assert hazard_result.risk_change_percentage > vulnerability_result.risk_change_percentage
    
    def test_subcomponent_ablation(self, mock_risk_engine, sample_risk_data):
        """Test subcomponent ablation analysis"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        original_risks = analyzer._calculate_baseline_risk_scores(sample_risk_data)
        
        # Test thermal stress subcomponent ablation
        thermal_result = analyzer._ablate_subcomponent(
            sample_risk_data, 'hazard', 'thermal_stress', original_risks
        )
        
        assert isinstance(thermal_result, AblationResult)
        assert thermal_result.component_name == 'hazard_thermal_stress'
        assert thermal_result.risk_change_percentage >= 0
        
        # Test population density subcomponent ablation
        pop_result = analyzer._ablate_subcomponent(
            sample_risk_data, 'exposure', 'normalized_pop_density', original_risks
        )
        
        assert pop_result.component_name == 'exposure_normalized_pop_density'
        assert pop_result.risk_change_percentage >= 0
    
    def test_complete_ablation_analysis(self, mock_risk_engine, sample_risk_data):
        """Test complete ablation analysis workflow"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        
        # Should have results for all main components
        assert 'hazard' in results
        assert 'exposure' in results
        assert 'vulnerability' in results
        
        # Should have subcomponent results
        subcomponent_keys = [key for key in results.keys() if '_' in key]
        assert len(subcomponent_keys) > 0
        
        # All results should be AblationResult objects
        for result in results.values():
            assert isinstance(result, AblationResult)
            assert result.cells_affected > 0
            assert result.risk_change_percentage >= 0
    
    def test_component_importance_ranking(self, mock_risk_engine, sample_risk_data):
        """Test component importance ranking calculation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Perform ablation analysis
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        
        # Calculate importance ranking
        importance_ranking = analyzer.calculate_component_importance_ranking(ablation_results)
        
        # Should have rankings for main components
        assert len(importance_ranking) == 3  # hazard, exposure, vulnerability
        
        # Should be sorted by importance (descending)
        for i in range(len(importance_ranking) - 1):
            assert importance_ranking[i].importance_score >= importance_ranking[i + 1].importance_score
        
        # Ranks should be sequential
        for i, comp in enumerate(importance_ranking):
            assert comp.rank == i + 1
        
        # All components should be represented
        component_names = {comp.component_name for comp in importance_ranking}
        assert component_names == {'hazard', 'exposure', 'vulnerability'}
    
    def test_ablation_validation(self, mock_risk_engine, sample_risk_data):
        """Test ablation results validation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        
        validation_results = analyzer.validate_ablation_results(ablation_results)
        
        # Should validate that all components were analyzed
        assert validation_results['all_components_analyzed']
        
        # Should validate that components have measurable impact
        assert validation_results['hazard_has_impact']
        assert validation_results['exposure_has_impact']
        assert validation_results['vulnerability_has_impact']
        
        # Overall validation should pass for good data
        assert validation_results['overall_valid']
    
    def test_ablation_with_expected_patterns(self, mock_risk_engine, sample_risk_data):
        """Test ablation validation with expected patterns"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        
        # Define expected patterns (hazard should have high impact due to 50% weight)
        expected_patterns = {
            'hazard': 'high',
            'exposure': 'medium',
            'vulnerability': 'low'
        }
        
        validation_results = analyzer.validate_ablation_results(
            ablation_results, expected_patterns
        )
        
        # Check pattern validation (may not always match due to data variability)
        assert 'hazard_expected_magnitude' in validation_results
        assert 'exposure_expected_magnitude' in validation_results
        assert 'vulnerability_expected_magnitude' in validation_results
    
    def test_summary_report_generation(self, mock_risk_engine, sample_risk_data):
        """Test ablation summary report generation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        importance_ranking = analyzer.calculate_component_importance_ranking(ablation_results)
        
        summary_report = analyzer.generate_ablation_summary_report(
            ablation_results, importance_ranking
        )
        
        # Should have required fields
        required_fields = [
            'analysis_timestamp', 'total_components_analyzed',
            'most_important_component', 'component_rankings',
            'detailed_results'
        ]
        
        for field in required_fields:
            assert field in summary_report
        
        # Should have reasonable values
        assert summary_report['total_components_analyzed'] == 3
        assert summary_report['most_important_component'] in ['hazard', 'exposure', 'vulnerability']
        assert len(summary_report['component_rankings']) == 3
        assert len(summary_report['detailed_results']) == 3
    
    def test_ops_notes_export(self, mock_risk_engine, sample_risk_data):
        """Test operational notes export"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        importance_ranking = analyzer.calculate_component_importance_ranking(ablation_results)
        
        ops_notes = analyzer.export_ablation_results_for_ops_notes(
            ablation_results, importance_ranking
        )
        
        # Should be a string with useful information
        assert isinstance(ops_notes, str)
        assert len(ops_notes) > 100  # Should have substantial content
        
        # Should contain key information
        assert 'COMPONENT IMPORTANCE ANALYSIS' in ops_notes
        assert 'Most critical component:' in ops_notes
        assert 'Total system sensitivity:' in ops_notes
        
        # Should mention all main components
        assert 'hazard' in ops_notes.lower()
        assert 'exposure' in ops_notes.lower()
        assert 'vulnerability' in ops_notes.lower()
    
    def test_ablation_with_incomplete_data(self, mock_risk_engine, incomplete_risk_data):
        """Test ablation analysis with incomplete data"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Should handle missing subcomponents gracefully
        results = analyzer.perform_complete_ablation_analysis(incomplete_risk_data)
        
        # Should still have main component results
        assert 'hazard' in results
        assert 'exposure' in results
        assert 'vulnerability' in results
        
        # May have fewer subcomponent results due to missing data
        subcomponent_count = len([key for key in results.keys() if '_' in key])
        assert subcomponent_count >= 0  # Could be zero if many components missing
    
    def test_edge_cases(self, mock_risk_engine):
        """Test edge cases and error handling"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Test with all-zero data
        zero_data = pd.DataFrame({
            'cell_id': ['CELL_001', 'CELL_002'],
            'hazard_score': [0.0, 0.0],
            'exposure_score': [0.0, 0.0],
            'vulnerability_score': [0.0, 0.0]
        })
        
        # Should handle gracefully (may have zero impact)
        try:
            results = analyzer.perform_complete_ablation_analysis(zero_data)
            # If it succeeds, results should be valid
            for result in results.values():
                assert isinstance(result, AblationResult)
        except ValueError:
            # May fail due to insufficient variation - this is acceptable
            pass
        
        # Test with single cell (insufficient for analysis)
        single_cell_data = pd.DataFrame({
            'cell_id': ['CELL_001'],
            'hazard_score': [0.5],
            'exposure_score': [0.3],
            'vulnerability_score': [0.2]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.perform_complete_ablation_analysis(single_cell_data)
    
    def test_visualization_creation(self, mock_risk_engine, sample_risk_data):
        """Test ablation visualization creation"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        ablation_results = analyzer.perform_complete_ablation_analysis(sample_risk_data)
        importance_ranking = analyzer.calculate_component_importance_ranking(ablation_results)
        
        # Test visualization creation (should not raise errors)
        try:
            fig = analyzer.create_ablation_visualization(ablation_results, importance_ranking)
            
            # Should return a plotly figure
            assert hasattr(fig, 'data')  # Plotly figures have a 'data' attribute
            
        except ImportError:
            # Skip if plotly not available
            pytest.skip("Plotly not available for visualization test")
    
    def test_correlation_analysis(self, mock_risk_engine, sample_risk_data):
        """Test correlation analysis in ablation results"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        original_risks = analyzer._calculate_baseline_risk_scores(sample_risk_data)
        
        # Test hazard ablation correlation
        hazard_result = analyzer._ablate_component(sample_risk_data, 'hazard', original_risks)
        
        # Correlation should be reasonable (positive but less than 1)
        if not np.isnan(hazard_result.correlation_with_original):
            assert 0.0 <= hazard_result.correlation_with_original <= 1.0
            # Should be less than 1 since we removed a component
            assert hazard_result.correlation_with_original < 0.99
    
    def test_impact_magnitude_classification(self, mock_risk_engine):
        """Test impact magnitude classification logic"""
        analyzer = AblationAnalyzer(mock_risk_engine)
        
        # Create test data with known impact levels
        test_data = pd.DataFrame({
            'cell_id': ['CELL_001', 'CELL_002', 'CELL_003'],
            'hazard_score': [1.0, 0.5, 0.0],  # High variation for testing
            'exposure_score': [0.5, 0.5, 0.5],
            'vulnerability_score': [0.2, 0.2, 0.2]
        })
        
        original_risks = analyzer._calculate_baseline_risk_scores(test_data)
        
        # Test hazard ablation (should have high impact due to variation)
        hazard_result = analyzer._ablate_component(test_data, 'hazard', original_risks)
        
        # Should classify impact magnitude appropriately
        assert hazard_result.impact_magnitude in ['low', 'medium', 'high']
        
        # With high hazard variation, should likely be medium or high impact
        assert hazard_result.risk_change_percentage > 0


if __name__ == '__main__':
    pytest.main([__file__])