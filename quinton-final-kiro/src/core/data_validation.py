"""
Data Coverage Validation System for MISO Weather-Stress Heatmap

This module implements comprehensive data quality validation and coverage assessment
to ensure reliable risk calculations and automatic degradation to demo mode when
data quality falls below acceptable thresholds.

Key Components:
- Weather data coverage validation
- Infrastructure data coverage validation  
- Automatic demo mode degradation
- Data quality issue logging and reporting
- Coverage statistics calculation
- User warning generation for data gaps

Requirements addressed: 6.2, 6.4
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class CoverageThresholds:
    """Configuration for minimum data coverage thresholds"""
    # Weather data thresholds
    min_weather_coverage: float = 0.7  # 70% of grid cells must have weather data
    min_weather_parameters: int = 6     # Minimum required weather parameters
    max_weather_age_hours: int = 6      # Maximum age of weather data in hours
    
    # Infrastructure data thresholds
    min_population_coverage: float = 0.8  # 80% of grid cells must have population data
    min_capacity_coverage: float = 0.5    # 50% of grid cells must have capacity data within 50km
    
    # Overall system thresholds
    min_overall_coverage: float = 0.6     # 60% overall data completeness for live mode
    max_missing_parameters: int = 2       # Maximum missing critical parameters


@dataclass
class DataQualityIssue:
    """Represents a data quality issue"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'weather', 'infrastructure', 'spatial', 'temporal'
    message: str
    affected_cells: int
    total_cells: int
    coverage_percentage: float
    timestamp: datetime


@dataclass
class CoverageReport:
    """Comprehensive data coverage assessment report"""
    overall_coverage: float
    weather_coverage: float
    infrastructure_coverage: float
    population_coverage: float
    capacity_coverage: float
    transmission_coverage: float
    
    total_cells: int
    cells_with_weather: int
    cells_with_infrastructure: int
    
    data_quality_issues: List[DataQualityIssue]
    recommendations: List[str]
    should_degrade_to_demo: bool
    confidence_penalty: float


class DataCoverageValidator:
    """
    Main data coverage validation system that assesses data quality,
    identifies gaps, and determines if degradation to demo mode is needed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data coverage validator.
        
        Args:
            config: Configuration dictionary with thresholds and parameters
        """
        self.config = config or self._get_default_config()
        self.thresholds = CoverageThresholds(**self.config.get('coverage_thresholds', {}))
        self.data_quality_issues = []
        
        logging.info("Data coverage validator initialized")
        logging.info(f"Weather coverage threshold: {self.thresholds.min_weather_coverage:.1%}")
        logging.info(f"Infrastructure coverage threshold: {self.thresholds.min_capacity_coverage:.1%}")
        logging.info(f"Overall coverage threshold: {self.thresholds.min_overall_coverage:.1%}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided"""
        return {
            'coverage_thresholds': {
                'min_weather_coverage': 0.7,
                'min_weather_parameters': 6,
                'max_weather_age_hours': 6,
                'min_population_coverage': 0.8,
                'min_capacity_coverage': 0.5,
                'min_overall_coverage': 0.6,
                'max_missing_parameters': 2
            }
        }
    
    def validate_weather_data_coverage(self, 
                                     weather_data: pd.DataFrame,
                                     grid: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate weather data coverage and quality.
        
        Args:
            weather_data: DataFrame with weather forecast data
            grid: GeoDataFrame with grid cells
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'coverage_percentage': 0.0,
            'parameter_completeness': {},
            'temporal_coverage': {},
            'data_freshness': {},
            'issues': []
        }
        
        total_cells = len(grid)
        
        if weather_data is None or len(weather_data) == 0:
            issue = DataQualityIssue(
                severity='critical',
                category='weather',
                message='No weather data available',
                affected_cells=total_cells,
                total_cells=total_cells,
                coverage_percentage=0.0,
                timestamp=datetime.now()
            )
            validation_results['issues'].append(issue)
            self.data_quality_issues.append(issue)
            return validation_results
        
        # Check spatial coverage
        unique_cells_with_weather = weather_data['cell_id'].nunique()
        weather_coverage = unique_cells_with_weather / total_cells
        validation_results['coverage_percentage'] = weather_coverage
        
        if weather_coverage < self.thresholds.min_weather_coverage:
            issue = DataQualityIssue(
                severity='critical' if weather_coverage < 0.5 else 'warning',
                category='weather',
                message=f'Weather data coverage below threshold: {weather_coverage:.1%} < {self.thresholds.min_weather_coverage:.1%}',
                affected_cells=total_cells - unique_cells_with_weather,
                total_cells=total_cells,
                coverage_percentage=weather_coverage,
                timestamp=datetime.now()
            )
            validation_results['issues'].append(issue)
            self.data_quality_issues.append(issue)
        
        # Check parameter completeness
        required_weather_params = [
            'temp_2m', 'wind_speed', 'wind_gust', 'precip_rate', 
            'relative_humidity', 'dewpoint'
        ]
        
        for param in required_weather_params:
            if param in weather_data.columns:
                non_null_count = weather_data[param].notna().sum()
                completeness = non_null_count / len(weather_data)
                validation_results['parameter_completeness'][param] = completeness
                
                if completeness < 0.8:  # 80% completeness threshold
                    issue = DataQualityIssue(
                        severity='warning' if completeness > 0.5 else 'critical',
                        category='weather',
                        message=f'Weather parameter {param} has low completeness: {completeness:.1%}',
                        affected_cells=len(weather_data) - non_null_count,
                        total_cells=len(weather_data),
                        coverage_percentage=completeness,
                        timestamp=datetime.now()
                    )
                    validation_results['issues'].append(issue)
                    self.data_quality_issues.append(issue)
            else:
                validation_results['parameter_completeness'][param] = 0.0
                issue = DataQualityIssue(
                    severity='critical',
                    category='weather',
                    message=f'Required weather parameter {param} is missing',
                    affected_cells=len(weather_data),
                    total_cells=len(weather_data),
                    coverage_percentage=0.0,
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        # Check temporal coverage (forecast horizons)
        if 'horizon_h' in weather_data.columns:
            available_horizons = set(weather_data['horizon_h'].unique())
            expected_horizons = {12, 24, 36, 48}
            missing_horizons = expected_horizons - available_horizons
            
            validation_results['temporal_coverage'] = {
                'available_horizons': list(available_horizons),
                'missing_horizons': list(missing_horizons),
                'horizon_completeness': len(available_horizons) / len(expected_horizons)
            }
            
            if missing_horizons:
                issue = DataQualityIssue(
                    severity='warning',
                    category='weather',
                    message=f'Missing forecast horizons: {missing_horizons}',
                    affected_cells=0,  # Not cell-specific
                    total_cells=total_cells,
                    coverage_percentage=len(available_horizons) / len(expected_horizons),
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        # Check data freshness
        if 'timestamp' in weather_data.columns:
            try:
                latest_timestamp = pd.to_datetime(weather_data['timestamp']).max()
                data_age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
                
                validation_results['data_freshness'] = {
                    'latest_timestamp': latest_timestamp,
                    'age_hours': data_age_hours,
                    'is_fresh': data_age_hours <= self.thresholds.max_weather_age_hours
                }
                
                if data_age_hours > self.thresholds.max_weather_age_hours:
                    issue = DataQualityIssue(
                        severity='warning',
                        category='weather',
                        message=f'Weather data is stale: {data_age_hours:.1f} hours old > {self.thresholds.max_weather_age_hours} hours',
                        affected_cells=len(weather_data),
                        total_cells=len(weather_data),
                        coverage_percentage=1.0,  # All data is stale
                        timestamp=datetime.now()
                    )
                    validation_results['issues'].append(issue)
                    self.data_quality_issues.append(issue)
            except Exception as e:
                logging.warning(f"Could not parse weather timestamps: {e}")
        
        return validation_results
    
    def validate_infrastructure_data_coverage(self,
                                            infrastructure_data: pd.DataFrame,
                                            grid: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate infrastructure data coverage and quality.
        
        Args:
            infrastructure_data: DataFrame with infrastructure data
            grid: GeoDataFrame with grid cells
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'population_coverage': 0.0,
            'capacity_coverage': 0.0,
            'transmission_coverage': 0.0,
            'parameter_completeness': {},
            'issues': []
        }
        
        total_cells = len(grid)
        
        if infrastructure_data is None or len(infrastructure_data) == 0:
            issue = DataQualityIssue(
                severity='critical',
                category='infrastructure',
                message='No infrastructure data available',
                affected_cells=total_cells,
                total_cells=total_cells,
                coverage_percentage=0.0,
                timestamp=datetime.now()
            )
            validation_results['issues'].append(issue)
            self.data_quality_issues.append(issue)
            return validation_results
        
        # Check population data coverage
        if 'total_population' in infrastructure_data.columns:
            cells_with_population = (infrastructure_data['total_population'] > 0).sum()
            population_coverage = cells_with_population / total_cells
            validation_results['population_coverage'] = population_coverage
            
            if population_coverage < self.thresholds.min_population_coverage:
                issue = DataQualityIssue(
                    severity='warning',
                    category='infrastructure',
                    message=f'Population data coverage below threshold: {population_coverage:.1%} < {self.thresholds.min_population_coverage:.1%}',
                    affected_cells=total_cells - cells_with_population,
                    total_cells=total_cells,
                    coverage_percentage=population_coverage,
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        # Check capacity data coverage
        if 'total_capacity_mw' in infrastructure_data.columns:
            cells_with_capacity = (infrastructure_data['total_capacity_mw'] > 0).sum()
            capacity_coverage = cells_with_capacity / total_cells
            validation_results['capacity_coverage'] = capacity_coverage
            
            if capacity_coverage < self.thresholds.min_capacity_coverage:
                issue = DataQualityIssue(
                    severity='warning',
                    category='infrastructure',
                    message=f'Generation capacity coverage below threshold: {capacity_coverage:.1%} < {self.thresholds.min_capacity_coverage:.1%}',
                    affected_cells=total_cells - cells_with_capacity,
                    total_cells=total_cells,
                    coverage_percentage=capacity_coverage,
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        # Check transmission data coverage
        if 'transmission_density' in infrastructure_data.columns:
            # Check if using baseline values (0.5) vs actual data
            baseline_mask = np.isclose(infrastructure_data['transmission_density'], 0.5, atol=1e-6)
            cells_with_actual_tx_data = (~baseline_mask).sum()
            transmission_coverage = cells_with_actual_tx_data / total_cells
            validation_results['transmission_coverage'] = transmission_coverage
            
            if transmission_coverage == 0:
                issue = DataQualityIssue(
                    severity='info',
                    category='infrastructure',
                    message='Using baseline transmission scarcity values (no actual transmission data)',
                    affected_cells=total_cells,
                    total_cells=total_cells,
                    coverage_percentage=0.0,
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        # Check parameter completeness for critical infrastructure metrics
        required_infra_params = [
            'normalized_pop_density', 'renewable_share', 'transmission_scarcity'
        ]
        
        for param in required_infra_params:
            if param in infrastructure_data.columns:
                non_null_count = infrastructure_data[param].notna().sum()
                completeness = non_null_count / len(infrastructure_data)
                validation_results['parameter_completeness'][param] = completeness
                
                if completeness < 0.9:  # 90% completeness threshold for infrastructure
                    issue = DataQualityIssue(
                        severity='warning' if completeness > 0.7 else 'critical',
                        category='infrastructure',
                        message=f'Infrastructure parameter {param} has low completeness: {completeness:.1%}',
                        affected_cells=len(infrastructure_data) - non_null_count,
                        total_cells=len(infrastructure_data),
                        coverage_percentage=completeness,
                        timestamp=datetime.now()
                    )
                    validation_results['issues'].append(issue)
                    self.data_quality_issues.append(issue)
            else:
                validation_results['parameter_completeness'][param] = 0.0
                issue = DataQualityIssue(
                    severity='critical',
                    category='infrastructure',
                    message=f'Required infrastructure parameter {param} is missing',
                    affected_cells=len(infrastructure_data),
                    total_cells=len(infrastructure_data),
                    coverage_percentage=0.0,
                    timestamp=datetime.now()
                )
                validation_results['issues'].append(issue)
                self.data_quality_issues.append(issue)
        
        return validation_results
    
    def assess_overall_coverage(self,
                              weather_validation: Dict[str, Any],
                              infrastructure_validation: Dict[str, Any]) -> CoverageReport:
        """
        Assess overall data coverage and determine if demo mode degradation is needed.
        
        Args:
            weather_validation: Weather data validation results
            infrastructure_validation: Infrastructure data validation results
            
        Returns:
            Comprehensive coverage report with recommendations
        """
        # Calculate overall coverage metrics
        weather_coverage = weather_validation.get('coverage_percentage', 0.0)
        population_coverage = infrastructure_validation.get('population_coverage', 0.0)
        capacity_coverage = infrastructure_validation.get('capacity_coverage', 0.0)
        transmission_coverage = infrastructure_validation.get('transmission_coverage', 0.0)
        
        # Calculate weighted overall coverage
        # Weather data is most critical, followed by population, then capacity
        overall_coverage = (
            0.5 * weather_coverage +
            0.3 * population_coverage +
            0.2 * capacity_coverage
        )
        
        # Collect all issues
        all_issues = []
        all_issues.extend(weather_validation.get('issues', []))
        all_issues.extend(infrastructure_validation.get('issues', []))
        
        # Count critical issues
        critical_issues = [issue for issue in all_issues if issue.severity == 'critical']
        
        # Determine if degradation to demo mode is needed
        should_degrade = (
            overall_coverage < self.thresholds.min_overall_coverage or
            len(critical_issues) > self.thresholds.max_missing_parameters or
            weather_coverage < 0.3  # Absolute minimum for weather data
        )
        
        # Calculate confidence penalty based on data gaps
        confidence_penalty = self._calculate_confidence_penalty(
            weather_coverage, population_coverage, capacity_coverage, len(critical_issues)
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            weather_coverage, population_coverage, capacity_coverage, 
            transmission_coverage, critical_issues
        )
        
        # Create coverage report
        report = CoverageReport(
            overall_coverage=overall_coverage,
            weather_coverage=weather_coverage,
            infrastructure_coverage=(population_coverage + capacity_coverage) / 2,
            population_coverage=population_coverage,
            capacity_coverage=capacity_coverage,
            transmission_coverage=transmission_coverage,
            total_cells=0,  # Will be set by caller
            cells_with_weather=0,  # Will be set by caller
            cells_with_infrastructure=0,  # Will be set by caller
            data_quality_issues=all_issues,
            recommendations=recommendations,
            should_degrade_to_demo=should_degrade,
            confidence_penalty=confidence_penalty
        )
        
        return report
    
    def _calculate_confidence_penalty(self,
                                    weather_coverage: float,
                                    population_coverage: float,
                                    capacity_coverage: float,
                                    critical_issue_count: int) -> float:
        """
        Calculate confidence penalty based on data coverage gaps.
        
        Returns:
            Confidence penalty factor [0,1] where 0 = no penalty, 1 = maximum penalty
        """
        # Base penalty from coverage gaps
        weather_penalty = max(0, (0.8 - weather_coverage) / 0.8) * 0.4  # Up to 40% penalty
        population_penalty = max(0, (0.8 - population_coverage) / 0.8) * 0.2  # Up to 20% penalty
        capacity_penalty = max(0, (0.5 - capacity_coverage) / 0.5) * 0.2  # Up to 20% penalty
        
        # Additional penalty for critical issues
        critical_penalty = min(critical_issue_count * 0.1, 0.3)  # Up to 30% penalty
        
        total_penalty = weather_penalty + population_penalty + capacity_penalty + critical_penalty
        
        return min(total_penalty, 0.8)  # Cap at 80% penalty
    
    def _generate_recommendations(self,
                                weather_coverage: float,
                                population_coverage: float,
                                capacity_coverage: float,
                                transmission_coverage: float,
                                critical_issues: List[DataQualityIssue]) -> List[str]:
        """Generate actionable recommendations based on data coverage assessment"""
        recommendations = []
        
        if weather_coverage < 0.7:
            recommendations.append(
                f"Weather data coverage is low ({weather_coverage:.1%}). "
                "Consider checking API connectivity or switching to fallback weather source."
            )
        
        if population_coverage < 0.8:
            recommendations.append(
                f"Population data coverage is low ({population_coverage:.1%}). "
                "Consider using regional population estimates for missing areas."
            )
        
        if capacity_coverage < 0.5:
            recommendations.append(
                f"Generation capacity data coverage is low ({capacity_coverage:.1%}). "
                "Consider expanding the search radius or using regional capacity estimates."
            )
        
        if transmission_coverage == 0:
            recommendations.append(
                "No actual transmission data available. Using baseline transmission scarcity values. "
                "Results may be less accurate in areas with significant transmission constraints."
            )
        
        if len(critical_issues) > 0:
            recommendations.append(
                f"Found {len(critical_issues)} critical data issues. "
                "Consider running in demo mode until data quality improves."
            )
        
        if not recommendations:
            recommendations.append("Data coverage meets quality thresholds. System ready for live operation.")
        
        return recommendations
    
    def create_user_warnings(self, coverage_report: CoverageReport) -> List[str]:
        """
        Create user-friendly warnings about data gaps and their impact on confidence.
        
        Args:
            coverage_report: Coverage assessment report
            
        Returns:
            List of user warning messages
        """
        warnings = []
        
        if coverage_report.should_degrade_to_demo:
            warnings.append(
                "⚠️  DATA QUALITY WARNING: Automatically switching to demo mode due to insufficient data coverage. "
                f"Overall coverage: {coverage_report.overall_coverage:.1%}"
            )
        
        if coverage_report.weather_coverage < 0.8:
            warnings.append(
                f"🌤️  Weather data coverage is limited ({coverage_report.weather_coverage:.1%}). "
                "Risk scores may be less reliable in areas without recent weather data."
            )
        
        if coverage_report.population_coverage < 0.8:
            warnings.append(
                f"👥 Population data coverage is limited ({coverage_report.population_coverage:.1%}). "
                "Exposure scores may be less accurate in some areas."
            )
        
        if coverage_report.capacity_coverage < 0.5:
            warnings.append(
                f"⚡ Generation capacity data coverage is limited ({coverage_report.capacity_coverage:.1%}). "
                "Vulnerability scores may not reflect actual generation mix in all areas."
            )
        
        if coverage_report.transmission_coverage == 0:
            warnings.append(
                "🔌 Using baseline transmission scarcity values. "
                "Vulnerability scores may not reflect actual transmission constraints."
            )
        
        if coverage_report.confidence_penalty > 0.3:
            warnings.append(
                f"📊 Data gaps will reduce confidence scores by up to {coverage_report.confidence_penalty:.0%}. "
                "Consider the confidence metrics when interpreting results."
            )
        
        return warnings
    
    def log_coverage_statistics(self, coverage_report: CoverageReport):
        """
        Log comprehensive coverage statistics for monitoring and debugging.
        
        Args:
            coverage_report: Coverage assessment report
        """
        logging.info("=== DATA COVERAGE ASSESSMENT ===")
        logging.info(f"Overall coverage: {coverage_report.overall_coverage:.1%}")
        logging.info(f"Weather coverage: {coverage_report.weather_coverage:.1%}")
        logging.info(f"Population coverage: {coverage_report.population_coverage:.1%}")
        logging.info(f"Capacity coverage: {coverage_report.capacity_coverage:.1%}")
        logging.info(f"Transmission coverage: {coverage_report.transmission_coverage:.1%}")
        
        logging.info(f"Total data quality issues: {len(coverage_report.data_quality_issues)}")
        
        # Log issues by severity
        critical_count = sum(1 for issue in coverage_report.data_quality_issues if issue.severity == 'critical')
        warning_count = sum(1 for issue in coverage_report.data_quality_issues if issue.severity == 'warning')
        info_count = sum(1 for issue in coverage_report.data_quality_issues if issue.severity == 'info')
        
        logging.info(f"Critical issues: {critical_count}")
        logging.info(f"Warning issues: {warning_count}")
        logging.info(f"Info issues: {info_count}")
        
        if coverage_report.should_degrade_to_demo:
            logging.warning("DEGRADING TO DEMO MODE due to insufficient data coverage")
        
        logging.info(f"Confidence penalty: {coverage_report.confidence_penalty:.1%}")
        
        # Log specific critical issues
        for issue in coverage_report.data_quality_issues:
            if issue.severity == 'critical':
                logging.error(f"CRITICAL: {issue.message}")
        
        logging.info("=== END COVERAGE ASSESSMENT ===")
    
    def get_coverage_summary_for_export(self, coverage_report: CoverageReport) -> Dict[str, Any]:
        """
        Get coverage summary suitable for export to method card or ops notes.
        
        Args:
            coverage_report: Coverage assessment report
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'assessment_timestamp': datetime.now().isoformat(),
            'overall_coverage_percentage': round(coverage_report.overall_coverage * 100, 1),
            'weather_coverage_percentage': round(coverage_report.weather_coverage * 100, 1),
            'population_coverage_percentage': round(coverage_report.population_coverage * 100, 1),
            'capacity_coverage_percentage': round(coverage_report.capacity_coverage * 100, 1),
            'transmission_coverage_percentage': round(coverage_report.transmission_coverage * 100, 1),
            'demo_mode_required': coverage_report.should_degrade_to_demo,
            'confidence_penalty_percentage': round(coverage_report.confidence_penalty * 100, 1),
            'critical_issues_count': sum(1 for issue in coverage_report.data_quality_issues if issue.severity == 'critical'),
            'warning_issues_count': sum(1 for issue in coverage_report.data_quality_issues if issue.severity == 'warning'),
            'data_quality_grade': self._calculate_data_quality_grade(coverage_report),
            'recommendations': coverage_report.recommendations
        }
    
    def _calculate_data_quality_grade(self, coverage_report: CoverageReport) -> str:
        """Calculate letter grade for overall data quality"""
        score = coverage_report.overall_coverage
        
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'