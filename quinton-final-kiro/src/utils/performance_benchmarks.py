"""
Performance Benchmarks and Validation for MISO Weather-Stress Heatmap

This module provides comprehensive performance benchmarking and validation
against known scenarios to ensure the system meets performance requirements
and produces accurate results under various conditions.

Requirements addressed: 5.1, 7.4
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import json

from demo_data_generator import DemoDataGenerator
from feature_engineering import FeatureEngineeringEngine
from risk_scoring_engine import RiskScoringEngine
from extended_risk_integration import ExtendedRiskScoringEngine
from extensibility_framework import ExtensibilityManager


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking"""
    operation_name: str
    data_size: int
    processing_time: float
    memory_usage_mb: float
    records_per_second: float
    peak_memory_mb: float
    cpu_usage_percent: float


@dataclass
class ValidationScenario:
    """Validation scenario definition"""
    name: str
    description: str
    expected_patterns: Dict[str, Any]
    tolerance: float = 0.1


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for the MISO system.
    Tests scalability, memory usage, and processing speed across different
    data sizes and configurations.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize performance benchmark suite.
        
        Args:
            random_seed: Random seed for reproducible benchmarks
        """
        self.random_seed = random_seed
        self.generator = DemoDataGenerator(random_seed)
        self.results: List[PerformanceMetrics] = []
        
        logging.info("Performance benchmark suite initialized")
    
    def benchmark_data_generation(self, 
                                grid_sizes: List[int] = [100, 500, 1000, 2000, 5000]) -> List[PerformanceMetrics]:
        """
        Benchmark data generation performance across different grid sizes.
        
        Args:
            grid_sizes: List of grid sizes to test
            
        Returns:
            List of performance metrics
        """
        metrics = []
        
        for n_cells in grid_sizes:
            logging.info(f"Benchmarking data generation for {n_cells} cells")
            
            # Monitor system resources
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            start_cpu_times = process.cpu_times()
            
            # Generate data
            grid = self.generator.generate_hex_grid_demo(n_cells=n_cells)
            weather_data = self.generator.generate_weather_demo_data(grid)
            infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
            
            end_time = time.time()
            end_cpu_times = process.cpu_times()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            total_records = len(weather_data) + len(infrastructure_data)
            records_per_second = total_records / processing_time if processing_time > 0 else 0
            
            cpu_time_used = (end_cpu_times.user - start_cpu_times.user + 
                           end_cpu_times.system - start_cpu_times.system)
            cpu_usage_percent = (cpu_time_used / processing_time) * 100 if processing_time > 0 else 0
            
            metric = PerformanceMetrics(
                operation_name=f'data_generation_{n_cells}_cells',
                data_size=n_cells,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                records_per_second=records_per_second,
                peak_memory_mb=final_memory,
                cpu_usage_percent=cpu_usage_percent
            )
            
            metrics.append(metric)
            self.results.append(metric)
            
            logging.info(f"Generated {total_records} records in {processing_time:.2f}s "
                        f"({records_per_second:.0f} records/s)")
        
        return metrics
    
    def benchmark_feature_engineering(self, 
                                    grid_sizes: List[int] = [100, 500, 1000, 2000]) -> List[PerformanceMetrics]:
        """
        Benchmark feature engineering performance.
        
        Args:
            grid_sizes: List of grid sizes to test
            
        Returns:
            List of performance metrics
        """
        metrics = []
        feature_engine = FeatureEngineeringEngine()
        
        for n_cells in grid_sizes:
            logging.info(f"Benchmarking feature engineering for {n_cells} cells")
            
            # Generate test data
            grid = self.generator.generate_hex_grid_demo(n_cells=n_cells)
            weather_data = self.generator.generate_weather_demo_data(grid)
            
            # Monitor system resources
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            # Process features
            processed_weather = feature_engine.process_weather_features(weather_data)
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            records_per_second = len(weather_data) / processing_time if processing_time > 0 else 0
            
            metric = PerformanceMetrics(
                operation_name=f'feature_engineering_{n_cells}_cells',
                data_size=len(weather_data),
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                records_per_second=records_per_second,
                peak_memory_mb=final_memory,
                cpu_usage_percent=0  # Not measured for this benchmark
            )
            
            metrics.append(metric)
            self.results.append(metric)
            
            logging.info(f"Processed {len(weather_data)} weather records in {processing_time:.2f}s")
        
        return metrics
    
    def benchmark_risk_scoring(self, 
                             grid_sizes: List[int] = [100, 500, 1000, 2000]) -> List[PerformanceMetrics]:
        """
        Benchmark risk scoring performance.
        
        Args:
            grid_sizes: List of grid sizes to test
            
        Returns:
            List of performance metrics
        """
        metrics = []
        feature_engine = FeatureEngineeringEngine()
        risk_engine = RiskScoringEngine()
        
        for n_cells in grid_sizes:
            logging.info(f"Benchmarking risk scoring for {n_cells} cells")
            
            # Generate and process test data
            grid = self.generator.generate_hex_grid_demo(n_cells=n_cells)
            weather_data = self.generator.generate_weather_demo_data(grid)
            infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
            processed_weather = feature_engine.process_weather_features(weather_data)
            
            # Monitor system resources
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            # Calculate risk scores
            hazard_data = risk_engine.process_hazard_scores(processed_weather)
            exposure_data = risk_engine.process_exposure_scores(infrastructure_data)
            vulnerability_data = risk_engine.process_vulnerability_scores(infrastructure_data)
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            total_records = len(hazard_data) + len(exposure_data) + len(vulnerability_data)
            records_per_second = total_records / processing_time if processing_time > 0 else 0
            
            metric = PerformanceMetrics(
                operation_name=f'risk_scoring_{n_cells}_cells',
                data_size=total_records,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                records_per_second=records_per_second,
                peak_memory_mb=final_memory,
                cpu_usage_percent=0  # Not measured for this benchmark
            )
            
            metrics.append(metric)
            self.results.append(metric)
            
            logging.info(f"Calculated risk scores for {total_records} records in {processing_time:.2f}s")
        
        return metrics
    
    def benchmark_extended_system(self, 
                                grid_sizes: List[int] = [100, 500, 1000]) -> List[PerformanceMetrics]:
        """
        Benchmark extended system with plugins.
        
        Args:
            grid_sizes: List of grid sizes to test
            
        Returns:
            List of performance metrics
        """
        metrics = []
        
        # Configure extended system
        extended_config = {
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
        
        for n_cells in grid_sizes:
            logging.info(f"Benchmarking extended system for {n_cells} cells")
            
            # Generate test data with extended columns
            grid = self.generator.generate_hex_grid_demo(n_cells=n_cells)
            weather_data = self.generator.generate_weather_demo_data(grid)
            infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
            
            # Add extended data columns
            infrastructure_data['renewable_transition_rate'] = np.random.uniform(0.02, 0.12, len(infrastructure_data))
            infrastructure_data['projected_load_growth_rate'] = np.random.uniform(0.01, 0.08, len(infrastructure_data))
            
            # Process features
            feature_engine = FeatureEngineeringEngine()
            processed_weather = feature_engine.process_weather_features(weather_data)
            
            # Monitor system resources
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            # Run extended system
            extended_engine = ExtendedRiskScoringEngine(extended_config)
            extended_hazard = extended_engine.calculate_extended_hazard_score(processed_weather)
            extended_exposure = extended_engine.calculate_extended_exposure_score(infrastructure_data)
            extended_vulnerability = extended_engine.calculate_extended_vulnerability_score(infrastructure_data)
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            total_records = len(extended_hazard) + len(extended_exposure) + len(extended_vulnerability)
            records_per_second = total_records / processing_time if processing_time > 0 else 0
            
            metric = PerformanceMetrics(
                operation_name=f'extended_system_{n_cells}_cells',
                data_size=total_records,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                records_per_second=records_per_second,
                peak_memory_mb=final_memory,
                cpu_usage_percent=0  # Not measured for this benchmark
            )
            
            metrics.append(metric)
            self.results.append(metric)
            
            logging.info(f"Extended system processed {total_records} records in {processing_time:.2f}s")
        
        return metrics
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[PerformanceMetrics]]:
        """
        Run comprehensive performance benchmark suite.
        
        Returns:
            Dictionary of benchmark results by category
        """
        logging.info("Starting comprehensive performance benchmark suite")
        
        results = {
            'data_generation': self.benchmark_data_generation([100, 500, 1000, 2000]),
            'feature_engineering': self.benchmark_feature_engineering([100, 500, 1000]),
            'risk_scoring': self.benchmark_risk_scoring([100, 500, 1000]),
            'extended_system': self.benchmark_extended_system([100, 500])
        }
        
        logging.info("Comprehensive benchmark suite completed")
        return results
    
    def analyze_scalability(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """
        Analyze scalability characteristics from performance metrics.
        
        Args:
            metrics: List of performance metrics
            
        Returns:
            Dictionary of scalability analysis results
        """
        if len(metrics) < 2:
            return {'error': 'Insufficient data for scalability analysis'}
        
        # Sort by data size
        sorted_metrics = sorted(metrics, key=lambda x: x.data_size)
        
        # Calculate scaling factors
        size_ratios = []
        time_ratios = []
        memory_ratios = []
        
        for i in range(1, len(sorted_metrics)):
            prev_metric = sorted_metrics[i-1]
            curr_metric = sorted_metrics[i]
            
            size_ratio = curr_metric.data_size / prev_metric.data_size
            time_ratio = curr_metric.processing_time / prev_metric.processing_time
            memory_ratio = curr_metric.memory_usage_mb / prev_metric.memory_usage_mb if prev_metric.memory_usage_mb > 0 else 1.0
            
            size_ratios.append(size_ratio)
            time_ratios.append(time_ratio)
            memory_ratios.append(memory_ratio)
        
        # Calculate average scaling characteristics
        avg_size_ratio = np.mean(size_ratios)
        avg_time_ratio = np.mean(time_ratios)
        avg_memory_ratio = np.mean(memory_ratios)
        
        # Estimate complexity (log-log slope)
        sizes = [m.data_size for m in sorted_metrics]
        times = [m.processing_time for m in sorted_metrics]
        
        if len(sizes) >= 2:
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            complexity_slope = np.polyfit(log_sizes, log_times, 1)[0]
        else:
            complexity_slope = 1.0
        
        return {
            'avg_size_scaling': avg_size_ratio,
            'avg_time_scaling': avg_time_ratio,
            'avg_memory_scaling': avg_memory_ratio,
            'complexity_estimate': complexity_slope,
            'linear_scaling': abs(complexity_slope - 1.0) < 0.2,
            'sublinear_scaling': complexity_slope < 0.8,
            'superlinear_scaling': complexity_slope > 1.2
        }
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Performance report dictionary
        """
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            'performance_metrics': [],
            'scalability_analysis': {},
            'recommendations': []
        }
        
        # Add all performance metrics
        for metric in self.results:
            report['performance_metrics'].append({
                'operation': metric.operation_name,
                'data_size': metric.data_size,
                'processing_time_s': metric.processing_time,
                'memory_usage_mb': metric.memory_usage_mb,
                'records_per_second': metric.records_per_second,
                'peak_memory_mb': metric.peak_memory_mb
            })
        
        # Group metrics by operation type for scalability analysis
        operation_groups = {}
        for metric in self.results:
            op_type = metric.operation_name.split('_')[0] + '_' + metric.operation_name.split('_')[1]
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(metric)
        
        # Analyze scalability for each operation type
        for op_type, metrics in operation_groups.items():
            if len(metrics) >= 2:
                scalability = self.analyze_scalability(metrics)
                report['scalability_analysis'][op_type] = scalability
        
        # Generate recommendations
        recommendations = []
        
        # Check for performance issues
        slow_operations = [m for m in self.results if m.records_per_second < 100]
        if slow_operations:
            recommendations.append(
                f"Performance concern: {len(slow_operations)} operations processing < 100 records/second"
            )
        
        # Check for memory issues
        high_memory_operations = [m for m in self.results if m.memory_usage_mb > 500]
        if high_memory_operations:
            recommendations.append(
                f"Memory concern: {len(high_memory_operations)} operations using > 500MB"
            )
        
        # Check scalability
        for op_type, scalability in report['scalability_analysis'].items():
            if scalability.get('superlinear_scaling', False):
                recommendations.append(
                    f"Scalability concern: {op_type} shows superlinear scaling (complexity: {scalability['complexity_estimate']:.2f})"
                )
        
        if not recommendations:
            recommendations.append("All performance metrics within acceptable ranges")
        
        report['recommendations'] = recommendations
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Performance report saved to {output_file}")
        
        return report


class ValidationSuite:
    """
    Validation suite for testing system accuracy against known scenarios.
    Ensures the system produces expected results under controlled conditions.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize validation suite.
        
        Args:
            random_seed: Random seed for reproducible validation
        """
        self.random_seed = random_seed
        self.generator = DemoDataGenerator(random_seed)
        
        # Define validation scenarios
        self.scenarios = [
            ValidationScenario(
                name='extreme_heat',
                description='Extreme heat wave conditions should produce high thermal stress',
                expected_patterns={
                    'thermal_stress_mean': (0.7, 1.0),
                    'hazard_score_mean': (0.5, 1.0),
                    'high_risk_cells_pct': (0.3, 1.0)
                }
            ),
            ValidationScenario(
                name='severe_wind',
                description='High wind conditions should produce elevated wind stress',
                expected_patterns={
                    'wind_stress_mean': (0.6, 1.0),
                    'storm_proxy_mean': (0.4, 1.0),
                    'hazard_score_mean': (0.4, 1.0)
                }
            ),
            ValidationScenario(
                name='urban_exposure',
                description='Urban areas should show higher exposure scores',
                expected_patterns={
                    'urban_exposure_higher': True,
                    'exposure_correlation': (0.5, 1.0)
                }
            ),
            ValidationScenario(
                name='renewable_vulnerability',
                description='High renewable areas should show specific vulnerability patterns',
                expected_patterns={
                    'renewable_vulnerability_correlation': (0.3, 1.0),
                    'vulnerability_range': (0.0, 1.0)
                }
            )
        ]
        
        logging.info("Validation suite initialized with {} scenarios".format(len(self.scenarios)))
    
    def validate_extreme_heat_scenario(self) -> Dict[str, Any]:
        """Validate system response to extreme heat conditions"""
        # Generate extreme heat scenario
        grid = self.generator.generate_hex_grid_demo(n_cells=100)
        weather_data = self.generator.generate_weather_demo_data(grid, scenario='heat_wave')
        
        # Process through system
        feature_engine = FeatureEngineeringEngine()
        processed_weather = feature_engine.process_weather_features(weather_data)
        
        risk_engine = RiskScoringEngine()
        hazard_data = risk_engine.process_hazard_scores(processed_weather)
        
        # Analyze results
        thermal_stress_mean = processed_weather['thermal_stress'].mean()
        hazard_score_mean = hazard_data['hazard_score'].mean()
        high_risk_cells = (hazard_data['hazard_score'] > 0.7).sum() / len(hazard_data)
        
        # Check against expected patterns
        scenario = next(s for s in self.scenarios if s.name == 'extreme_heat')
        results = {
            'scenario_name': scenario.name,
            'description': scenario.description,
            'metrics': {
                'thermal_stress_mean': thermal_stress_mean,
                'hazard_score_mean': hazard_score_mean,
                'high_risk_cells_pct': high_risk_cells
            },
            'validations': {}
        }
        
        # Validate each expected pattern
        for pattern_name, expected_range in scenario.expected_patterns.items():
            if pattern_name in results['metrics']:
                actual_value = results['metrics'][pattern_name]
                if isinstance(expected_range, tuple):
                    is_valid = expected_range[0] <= actual_value <= expected_range[1]
                else:
                    is_valid = actual_value == expected_range
                
                results['validations'][pattern_name] = {
                    'expected': expected_range,
                    'actual': actual_value,
                    'valid': is_valid
                }
        
        return results
    
    def validate_urban_exposure_scenario(self) -> Dict[str, Any]:
        """Validate urban vs rural exposure patterns"""
        # Generate data with urban clustering
        grid = self.generator.generate_hex_grid_demo(n_cells=200)
        infrastructure_data = self.generator.generate_infrastructure_demo_data(grid)
        
        # Process through system
        risk_engine = RiskScoringEngine()
        exposure_data = risk_engine.process_exposure_scores(infrastructure_data)
        
        # Classify urban vs rural areas
        urban_threshold = 0.7
        rural_threshold = 0.3
        
        urban_mask = infrastructure_data['normalized_pop_density'] >= urban_threshold
        rural_mask = infrastructure_data['normalized_pop_density'] <= rural_threshold
        
        urban_exposure = exposure_data[urban_mask]['exposure_score'].mean() if urban_mask.any() else 0
        rural_exposure = exposure_data[rural_mask]['exposure_score'].mean() if rural_mask.any() else 0
        
        # Calculate correlation
        correlation = infrastructure_data['normalized_pop_density'].corr(exposure_data['exposure_score'])
        
        # Analyze results
        results = {
            'scenario_name': 'urban_exposure',
            'description': 'Urban areas should show higher exposure scores',
            'metrics': {
                'urban_exposure_mean': urban_exposure,
                'rural_exposure_mean': rural_exposure,
                'urban_exposure_higher': urban_exposure > rural_exposure,
                'exposure_correlation': correlation
            },
            'validations': {}
        }
        
        # Validate patterns
        scenario = next(s for s in self.scenarios if s.name == 'urban_exposure')
        for pattern_name, expected in scenario.expected_patterns.items():
            if pattern_name in results['metrics']:
                actual = results['metrics'][pattern_name]
                if isinstance(expected, tuple):
                    is_valid = expected[0] <= actual <= expected[1]
                else:
                    is_valid = actual == expected
                
                results['validations'][pattern_name] = {
                    'expected': expected,
                    'actual': actual,
                    'valid': is_valid
                }
        
        return results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """
        Run all validation scenarios.
        
        Returns:
            Comprehensive validation results
        """
        logging.info("Running comprehensive validation suite")
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'scenarios': [],
            'overall_summary': {
                'total_scenarios': len(self.scenarios),
                'passed_scenarios': 0,
                'failed_scenarios': 0,
                'overall_success_rate': 0.0
            }
        }
        
        # Run individual validations
        scenario_results = [
            self.validate_extreme_heat_scenario(),
            self.validate_urban_exposure_scenario()
        ]
        
        # Analyze overall results
        passed_scenarios = 0
        for result in scenario_results:
            validation_results['scenarios'].append(result)
            
            # Check if all validations passed for this scenario
            all_valid = all(v['valid'] for v in result['validations'].values())
            if all_valid:
                passed_scenarios += 1
        
        validation_results['overall_summary']['passed_scenarios'] = passed_scenarios
        validation_results['overall_summary']['failed_scenarios'] = len(scenario_results) - passed_scenarios
        validation_results['overall_summary']['overall_success_rate'] = passed_scenarios / len(scenario_results)
        
        logging.info(f"Validation complete: {passed_scenarios}/{len(scenario_results)} scenarios passed")
        
        return validation_results


def run_complete_benchmark_and_validation(output_dir: str = 'benchmarks') -> Dict[str, Any]:
    """
    Run complete benchmark and validation suite.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        Combined results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run performance benchmarks
    benchmark_suite = PerformanceBenchmarkSuite()
    benchmark_results = benchmark_suite.run_comprehensive_benchmark()
    performance_report = benchmark_suite.generate_performance_report(
        os.path.join(output_dir, 'performance_report.json')
    )
    
    # Run validation suite
    validation_suite = ValidationSuite()
    validation_results = validation_suite.run_all_validations()
    
    # Save validation results
    with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Combine results
    combined_results = {
        'benchmark_results': benchmark_results,
        'performance_report': performance_report,
        'validation_results': validation_results,
        'summary': {
            'benchmark_operations': len(benchmark_suite.results),
            'validation_scenarios': validation_results['overall_summary']['total_scenarios'],
            'validation_success_rate': validation_results['overall_summary']['overall_success_rate'],
            'performance_concerns': len(performance_report.get('recommendations', [])),
            'overall_system_health': 'good' if validation_results['overall_summary']['overall_success_rate'] > 0.8 else 'needs_attention'
        }
    }
    
    # Save combined results
    with open(os.path.join(output_dir, 'combined_results.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logging.info(f"Complete benchmark and validation results saved to {output_dir}")
    return combined_results


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run complete benchmark and validation
    results = run_complete_benchmark_and_validation()
    
    # Print summary
    print("\n" + "="*60)
    print("MISO Weather-Stress Heatmap - Performance & Validation Summary")
    print("="*60)
    print(f"Benchmark Operations: {results['summary']['benchmark_operations']}")
    print(f"Validation Scenarios: {results['summary']['validation_scenarios']}")
    print(f"Validation Success Rate: {results['summary']['validation_success_rate']:.1%}")
    print(f"Performance Concerns: {results['summary']['performance_concerns']}")
    print(f"Overall System Health: {results['summary']['overall_system_health'].upper()}")
    print("="*60)