"""
Tests for Data Coverage Validation System

This module contains comprehensive tests for the data coverage validation
system, including coverage threshold checks, demo mode degradation logic,
and data quality issue detection.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
import logging

from data_validation import (
    DataCoverageValidator, CoverageThresholds, DataQualityIssue, 
    CoverageReport
)


class TestDataCoverageValidator:
    """Test suite for DataCoverageValidator class"""
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample grid for testing"""
        # Create 10x10 grid of cells
        cells = []
        for i in range(10):
            for j in range(10):
                cell_id = f"CELL_{i:02d}_{j:02d}"
                lon = -90.0 + i * 0.1
                lat = 40.0 + j * 0.1
                
                # Create square cell
                cell_polygon = Polygon([
                    (lon, lat), (lon + 0.1, lat),
                    (lon + 0.1, lat + 0.1), (lon, lat + 0.1),
                    (lon, lat)
                ])
                
                cells.append({
                    'cell_id': cell_id,
                    'centroid_lon': lon + 0.05,
                    'centroid_lat': lat + 0.05,
                    'area_km2': 100.0,  # Approximate
                    'geometry': cell_polygon
                })
        
        return gpd.GeoDataFrame(cells, crs='EPSG:4326')
    
    @pytest.fixture
    def complete_weather_data(self, sample_grid):
        """Create complete weather data for testing"""
        weather_records = []
        
        for _, cell in sample_grid.iterrows():
            for horizon in [12, 24, 36, 48]:
                weather_records.append({
                    'cell_id': cell['cell_id'],
                    'horizon_h': horizon,
                    'timestamp': datetime.now() + timedelta(hours=horizon),
                    'temp_2m': np.random.uniform(60, 90),
                    'wind_speed': np.random.uniform(5, 25),
                    'wind_gust': np.random.uniform(10, 35),
                    'precip_rate': np.random.uniform(0, 5),
                    'relative_humidity': np.random.uniform(40, 80),
                    'dewpoint': np.random.uniform(50, 70),
                    'confidence': 0.8
                })
        
        return pd.DataFrame(weather_records)
    
    @pytest.fixture
    def incomplete_weather_data(self, sample_grid):
        """Create incomplete weather data for testing"""
        weather_records = []
        
        # Only include data for 60% of cells
        selected_cells = sample_grid.sample(n=60, random_state=42)
        
        for _, cell in selected_cells.iterrows():
            for horizon in [12, 24]:  # Missing 36h and 48h horizons
                weather_records.append({
                    'cell_id': cell['cell_id'],
                    'horizon_h': horizon,
                    'timestamp': datetime.now() + timedelta(hours=horizon),
                    'temp_2m': np.random.uniform(60, 90),
                    'wind_speed': np.random.uniform(5, 25),
                    # Missing wind_gust, precip_rate
                    'relative_humidity': np.random.uniform(40, 80),
                    # Missing dewpoint
                    'confidence': 0.6
                })
        
        return pd.DataFrame(weather_records)
    
    @pytest.fixture
    def complete_infrastructure_data(self, sample_grid):
        """Create complete infrastructure data for testing"""
        infra_records = []
        
        for _, cell in sample_grid.iterrows():
            infra_records.append({
                'cell_id': cell['cell_id'],
                'total_population': np.random.uniform(1000, 50000),
                'population_density_per_km2': np.random.uniform(10, 500),
                'normalized_pop_density': np.random.uniform(0.1, 1.0),
                'total_capacity_mw': np.random.uniform(0, 1000),
                'renewable_share': np.random.uniform(0.1, 0.8),
                'transmission_density': np.random.uniform(0.2, 0.9),
                'transmission_scarcity': np.random.uniform(0.1, 0.8),
                'load_factor': np.random.uniform(0.0, 1.0)
            })
        
        return pd.DataFrame(infra_records)
    
    @pytest.fixture
    def incomplete_infrastructure_data(self, sample_grid):
        """Create incomplete infrastructure data for testing"""
        infra_records = []
        
        # Only include data for 70% of cells
        selected_cells = sample_grid.sample(n=70, random_state=42)
        
        for _, cell in selected_cells.iterrows():
            record = {
                'cell_id': cell['cell_id'],
                'total_population': np.random.uniform(1000, 50000),
                'normalized_pop_density': np.random.uniform(0.1, 1.0),
                'renewable_share': np.random.uniform(0.1, 0.8),
                'transmission_scarcity': 0.5  # Baseline value
            }
            
            # Randomly missing capacity data for some cells
            if np.random.random() > 0.3:
                record['total_capacity_mw'] = 0.0  # No capacity data
            else:
                record['total_capacity_mw'] = np.random.uniform(100, 1000)
            
            infra_records.append(record)
        
        return pd.DataFrame(infra_records)
    
    def test_validator_initialization(self):
        """Test validator initialization with default and custom config"""
        # Test default initialization
        validator = DataCoverageValidator()
        assert validator.thresholds.min_weather_coverage == 0.7
        assert validator.thresholds.min_overall_coverage == 0.6
        
        # Test custom configuration
        custom_config = {
            'coverage_thresholds': {
                'min_weather_coverage': 0.8,
                'min_overall_coverage': 0.7
            }
        }
        validator = DataCoverageValidator(custom_config)
        assert validator.thresholds.min_weather_coverage == 0.8
        assert validator.thresholds.min_overall_coverage == 0.7
    
    def test_weather_data_validation_complete(self, sample_grid, complete_weather_data):
        """Test weather data validation with complete data"""
        validator = DataCoverageValidator()
        
        results = validator.validate_weather_data_coverage(complete_weather_data, sample_grid)
        
        # Should have 100% coverage
        assert results['coverage_percentage'] == 1.0
        
        # Should have good parameter completeness
        for param in ['temp_2m', 'wind_speed', 'relative_humidity']:
            assert results['parameter_completeness'][param] == 1.0
        
        # Should have complete temporal coverage
        assert set(results['temporal_coverage']['available_horizons']) == {12, 24, 36, 48}
        assert len(results['temporal_coverage']['missing_horizons']) == 0
        
        # Should have no critical issues
        critical_issues = [issue for issue in results['issues'] if issue.severity == 'critical']
        assert len(critical_issues) == 0
    
    def test_weather_data_validation_incomplete(self, sample_grid, incomplete_weather_data):
        """Test weather data validation with incomplete data"""
        validator = DataCoverageValidator()
        
        results = validator.validate_weather_data_coverage(incomplete_weather_data, sample_grid)
        
        # Should have 60% coverage (60 out of 100 cells)
        assert results['coverage_percentage'] == 0.6
        
        # Should have missing parameters
        assert 'wind_gust' not in incomplete_weather_data.columns or \
               results['parameter_completeness'].get('wind_gust', 0) < 1.0
        
        # Should have missing temporal coverage
        assert len(results['temporal_coverage']['missing_horizons']) > 0
        
        # Should have issues due to low coverage
        assert len(results['issues']) > 0
        
        # Should have critical or warning issues
        serious_issues = [issue for issue in results['issues'] 
                         if issue.severity in ['critical', 'warning']]
        assert len(serious_issues) > 0
    
    def test_weather_data_validation_empty(self, sample_grid):
        """Test weather data validation with no data"""
        validator = DataCoverageValidator()
        
        results = validator.validate_weather_data_coverage(pd.DataFrame(), sample_grid)
        
        # Should have 0% coverage
        assert results['coverage_percentage'] == 0.0
        
        # Should have critical issues
        critical_issues = [issue for issue in results['issues'] if issue.severity == 'critical']
        assert len(critical_issues) > 0
        
        # First issue should be about no data
        assert 'No weather data available' in critical_issues[0].message
    
    def test_infrastructure_data_validation_complete(self, sample_grid, complete_infrastructure_data):
        """Test infrastructure data validation with complete data"""
        validator = DataCoverageValidator()
        
        results = validator.validate_infrastructure_data_coverage(complete_infrastructure_data, sample_grid)
        
        # Should have good coverage
        assert results['population_coverage'] == 1.0
        assert results['capacity_coverage'] == 1.0
        
        # Should have good parameter completeness
        for param in ['normalized_pop_density', 'renewable_share']:
            assert results['parameter_completeness'][param] == 1.0
        
        # Should have minimal issues
        critical_issues = [issue for issue in results['issues'] if issue.severity == 'critical']
        assert len(critical_issues) == 0
    
    def test_infrastructure_data_validation_incomplete(self, sample_grid, incomplete_infrastructure_data):
        """Test infrastructure data validation with incomplete data"""
        validator = DataCoverageValidator()
        
        results = validator.validate_infrastructure_data_coverage(incomplete_infrastructure_data, sample_grid)
        
        # Should have reduced coverage
        assert results['population_coverage'] < 1.0
        assert results['capacity_coverage'] < 1.0
        
        # Should have issues
        assert len(results['issues']) > 0
    
    def test_overall_coverage_assessment_good_data(self, sample_grid, complete_weather_data, complete_infrastructure_data):
        """Test overall coverage assessment with good data"""
        validator = DataCoverageValidator()
        
        weather_validation = validator.validate_weather_data_coverage(complete_weather_data, sample_grid)
        infra_validation = validator.validate_infrastructure_data_coverage(complete_infrastructure_data, sample_grid)
        
        report = validator.assess_overall_coverage(weather_validation, infra_validation)
        
        # Should have high overall coverage
        assert report.overall_coverage > 0.8
        
        # Should not need to degrade to demo mode
        assert not report.should_degrade_to_demo
        
        # Should have low confidence penalty
        assert report.confidence_penalty < 0.2
        
        # Should have positive recommendations
        assert any('ready for live operation' in rec.lower() for rec in report.recommendations)
    
    def test_overall_coverage_assessment_poor_data(self, sample_grid, incomplete_weather_data, incomplete_infrastructure_data):
        """Test overall coverage assessment with poor data"""
        validator = DataCoverageValidator()
        
        weather_validation = validator.validate_weather_data_coverage(incomplete_weather_data, sample_grid)
        infra_validation = validator.validate_infrastructure_data_coverage(incomplete_infrastructure_data, sample_grid)
        
        report = validator.assess_overall_coverage(weather_validation, infra_validation)
        
        # Should have lower overall coverage
        assert report.overall_coverage < 0.8
        
        # May need to degrade to demo mode
        if report.overall_coverage < 0.6:
            assert report.should_degrade_to_demo
        
        # Should have higher confidence penalty
        assert report.confidence_penalty > 0.1
        
        # Should have actionable recommendations
        assert len(report.recommendations) > 0
        assert any('coverage is low' in rec for rec in report.recommendations)
    
    def test_demo_mode_degradation_triggers(self, sample_grid):
        """Test various conditions that should trigger demo mode degradation"""
        validator = DataCoverageValidator()
        
        # Test 1: Very low weather coverage
        minimal_weather = pd.DataFrame([{
            'cell_id': 'CELL_00_00',
            'horizon_h': 12,
            'temp_2m': 70.0,
            'wind_speed': 10.0
        }])
        
        weather_val = validator.validate_weather_data_coverage(minimal_weather, sample_grid)
        infra_val = {'population_coverage': 0.8, 'capacity_coverage': 0.5, 'issues': []}
        
        report = validator.assess_overall_coverage(weather_val, infra_val)
        assert report.should_degrade_to_demo
        
        # Test 2: Multiple critical issues
        validator.data_quality_issues = [
            DataQualityIssue('critical', 'weather', 'Critical issue 1', 10, 100, 0.1, datetime.now()),
            DataQualityIssue('critical', 'weather', 'Critical issue 2', 20, 100, 0.2, datetime.now()),
            DataQualityIssue('critical', 'infrastructure', 'Critical issue 3', 30, 100, 0.3, datetime.now())
        ]
        
        weather_val = {'coverage_percentage': 0.7, 'issues': validator.data_quality_issues[:2]}
        infra_val = {'population_coverage': 0.8, 'capacity_coverage': 0.5, 'issues': validator.data_quality_issues[2:]}
        
        report = validator.assess_overall_coverage(weather_val, infra_val)
        assert report.should_degrade_to_demo
    
    def test_confidence_penalty_calculation(self):
        """Test confidence penalty calculation logic"""
        validator = DataCoverageValidator()
        
        # Test low weather coverage penalty
        penalty = validator._calculate_confidence_penalty(0.5, 0.8, 0.5, 0)
        assert penalty > 0.1  # Should have significant penalty
        
        # Test good coverage, no penalty
        penalty = validator._calculate_confidence_penalty(0.9, 0.9, 0.8, 0)
        assert penalty < 0.1  # Should have minimal penalty
        
        # Test critical issues penalty
        penalty = validator._calculate_confidence_penalty(0.8, 0.8, 0.8, 3)
        assert penalty > 0.2  # Should have penalty for critical issues
    
    def test_user_warnings_generation(self, sample_grid, incomplete_weather_data, incomplete_infrastructure_data):
        """Test user warning message generation"""
        validator = DataCoverageValidator()
        
        weather_validation = validator.validate_weather_data_coverage(incomplete_weather_data, sample_grid)
        infra_validation = validator.validate_infrastructure_data_coverage(incomplete_infrastructure_data, sample_grid)
        
        report = validator.assess_overall_coverage(weather_validation, infra_validation)
        warnings = validator.create_user_warnings(report)
        
        # Should have warnings
        assert len(warnings) > 0
        
        # Should have weather-related warnings if coverage is low
        if report.weather_coverage < 0.8:
            assert any('Weather data coverage' in warning for warning in warnings)
        
        # Should have demo mode warning if degradation is needed
        if report.should_degrade_to_demo:
            assert any('demo mode' in warning.lower() for warning in warnings)
    
    def test_coverage_statistics_logging(self, sample_grid, complete_weather_data, complete_infrastructure_data, caplog):
        """Test coverage statistics logging"""
        validator = DataCoverageValidator()
        
        weather_validation = validator.validate_weather_data_coverage(complete_weather_data, sample_grid)
        infra_validation = validator.validate_infrastructure_data_coverage(complete_infrastructure_data, sample_grid)
        
        report = validator.assess_overall_coverage(weather_validation, infra_validation)
        
        with caplog.at_level(logging.INFO):
            validator.log_coverage_statistics(report)
        
        # Should log coverage percentages
        assert any('Overall coverage:' in record.message for record in caplog.records)
        assert any('Weather coverage:' in record.message for record in caplog.records)
        
        # Should log issue counts
        assert any('Total data quality issues:' in record.message for record in caplog.records)
    
    def test_coverage_summary_export(self, sample_grid, complete_weather_data, complete_infrastructure_data):
        """Test coverage summary export for method card"""
        validator = DataCoverageValidator()
        
        weather_validation = validator.validate_weather_data_coverage(complete_weather_data, sample_grid)
        infra_validation = validator.validate_infrastructure_data_coverage(complete_infrastructure_data, sample_grid)
        
        report = validator.assess_overall_coverage(weather_validation, infra_validation)
        summary = validator.get_coverage_summary_for_export(report)
        
        # Should have required fields
        required_fields = [
            'assessment_timestamp', 'overall_coverage_percentage',
            'weather_coverage_percentage', 'demo_mode_required',
            'data_quality_grade', 'recommendations'
        ]
        
        for field in required_fields:
            assert field in summary
        
        # Should have reasonable values
        assert 0 <= summary['overall_coverage_percentage'] <= 100
        assert summary['data_quality_grade'] in ['A', 'B', 'C', 'D', 'F']
        assert isinstance(summary['recommendations'], list)
    
    def test_data_quality_grade_calculation(self):
        """Test data quality grade calculation"""
        validator = DataCoverageValidator()
        
        # Create mock reports with different coverage levels
        test_cases = [
            (0.95, 'A'),
            (0.85, 'B'),
            (0.75, 'C'),
            (0.65, 'D'),
            (0.45, 'F')
        ]
        
        for coverage, expected_grade in test_cases:
            mock_report = CoverageReport(
                overall_coverage=coverage,
                weather_coverage=coverage,
                infrastructure_coverage=coverage,
                population_coverage=coverage,
                capacity_coverage=coverage,
                transmission_coverage=coverage,
                total_cells=100,
                cells_with_weather=int(coverage * 100),
                cells_with_infrastructure=int(coverage * 100),
                data_quality_issues=[],
                recommendations=[],
                should_degrade_to_demo=coverage < 0.6,
                confidence_penalty=max(0, (0.8 - coverage) / 0.8)
            )
            
            grade = validator._calculate_data_quality_grade(mock_report)
            assert grade == expected_grade
    
    def test_stale_weather_data_detection(self, sample_grid):
        """Test detection of stale weather data"""
        validator = DataCoverageValidator()
        
        # Create stale weather data (8 hours old)
        stale_timestamp = datetime.now() - timedelta(hours=8)
        stale_weather = pd.DataFrame([{
            'cell_id': 'CELL_00_00',
            'horizon_h': 12,
            'timestamp': stale_timestamp,
            'temp_2m': 70.0,
            'wind_speed': 10.0
        }])
        
        results = validator.validate_weather_data_coverage(stale_weather, sample_grid)
        
        # Should detect stale data
        assert not results['data_freshness']['is_fresh']
        assert results['data_freshness']['age_hours'] > 6
        
        # Should have warning about stale data
        stale_issues = [issue for issue in results['issues'] if 'stale' in issue.message.lower()]
        assert len(stale_issues) > 0


if __name__ == '__main__':
    pytest.main([__file__])