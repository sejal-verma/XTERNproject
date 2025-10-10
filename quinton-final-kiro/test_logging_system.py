"""
Test suite for comprehensive logging and documentation system

Tests all components of the logging system including:
- Structured logging functionality
- Performance monitoring
- Configuration tracking
- Data source logging
- Documentation generation
- Reproducibility reporting
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from logging_system import (
    MISOLogger, DocumentationGenerator, LogEntry, PerformanceMetrics,
    DataSourceMetadata, initialize_logging, get_logger
)


class TestMISOLogger:
    """Test comprehensive logging functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MISOLogger(
            log_dir=self.temp_dir,
            log_level="DEBUG",
            enable_performance=True
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test logger initialization and setup"""
        assert self.logger.log_dir.exists()
        assert self.logger.log_level == 10  # DEBUG level
        assert self.logger.enable_performance is True
        assert len(self.logger.log_entries) >= 1  # Initialization log
        assert self.logger.session_id is not None
    
    def test_basic_logging_methods(self):
        """Test basic logging methods (debug, info, warning, error)"""
        initial_count = len(self.logger.log_entries)
        
        self.logger.debug("Debug message", component="test")
        self.logger.info("Info message", component="test")
        self.logger.warning("Warning message", component="test")
        self.logger.error("Error message", component="test")
        
        assert len(self.logger.log_entries) == initial_count + 4
        
        # Check log entry structure
        last_entry = self.logger.log_entries[-1]
        assert isinstance(last_entry, LogEntry)
        assert last_entry.level == "ERROR"
        assert last_entry.component == "test"
        assert last_entry.message == "Error message"
        assert last_entry.timestamp is not None
    
    def test_configuration_logging(self):
        """Test configuration parameter logging"""
        test_config = {
            'runtime': {
                'mode': 'demo',
                'horizons_h': [12, 24, 36, 48],
                'random_seed': 42
            },
            'weights': {
                'hazard': {'thermal': 0.3, 'wind': 0.3},
                'exposure': {'pop': 0.7, 'load': 0.3}
            }
        }
        
        initial_count = len(self.logger.configuration_log)
        self.logger.log_configuration(test_config, "test_config")
        
        assert len(self.logger.configuration_log) == initial_count + 1
        assert "test_config" in self.logger.configuration_log
        assert self.logger.configuration_log["test_config"]["config"] == test_config
    
    def test_data_source_logging(self):
        """Test data source metadata logging"""
        initial_count = len(self.logger.data_sources)
        
        self.logger.log_data_source(
            source_name="NOAA_NWS",
            url="https://api.weather.gov/gridpoints/test",
            data_type="weather_forecast",
            coverage_area="MISO",
            temporal_range="48h",
            api_version="v1.0",
            cache_status="fresh"
        )
        
        assert len(self.logger.data_sources) == initial_count + 1
        
        source = self.logger.data_sources[-1]
        assert isinstance(source, DataSourceMetadata)
        assert source.source_name == "NOAA_NWS"
        assert source.data_type == "weather_forecast"
        assert source.cache_status == "fresh"
    
    def test_performance_monitoring_context(self):
        """Test performance monitoring context manager"""
        initial_count = len(self.logger.performance_metrics)
        
        with self.logger.performance_monitor("test_operation", input_size=100):
            # Simulate some work
            time.sleep(0.01)
            data = [i**2 for i in range(1000)]
        
        assert len(self.logger.performance_metrics) == initial_count + 1
        
        metric = self.logger.performance_metrics[-1]
        assert isinstance(metric, PerformanceMetrics)
        assert metric.operation_name == "test_operation"
        assert metric.duration_ms > 0
        assert metric.input_size == 100
    
    def test_timing_decorator(self):
        """Test automatic timing decorator"""
        initial_count = len(self.logger.performance_metrics)
        
        @self.logger.timing_decorator("decorated_function")
        def test_function(data_size):
            return [i**2 for i in range(data_size)]
        
        result = test_function(500)
        
        assert len(result) == 500
        assert len(self.logger.performance_metrics) == initial_count + 1
        
        metric = self.logger.performance_metrics[-1]
        assert "decorated_function" in metric.operation_name
        assert metric.duration_ms > 0
    
    def test_reproducibility_report_generation(self):
        """Test comprehensive reproducibility report generation"""
        # Add some test data
        test_config = {'runtime': {'mode': 'test', 'seed': 42}}
        self.logger.log_configuration(test_config)
        
        self.logger.log_data_source(
            "test_source", "http://test.com", "test_data"
        )
        
        with self.logger.performance_monitor("test_op"):
            time.sleep(0.001)
        
        report = self.logger.generate_reproducibility_report()
        
        assert "reproducibility_report" in report
        report_data = report["reproducibility_report"]
        
        # Check required sections
        assert "system_info" in report_data
        assert "package_versions" in report_data
        assert "data_sources" in report_data
        assert "performance_summary" in report_data
        assert "session_summary" in report_data
        
        # Check system info
        system_info = report_data["system_info"]
        assert "platform" in system_info
        assert "python_version" in system_info
        
        # Check data sources
        assert len(report_data["data_sources"]) >= 1
        
        # Check performance summary
        perf_summary = report_data["performance_summary"]
        assert perf_summary["total_operations"] >= 1
        assert "operations" in perf_summary
    
    def test_log_export_functionality(self):
        """Test log export in multiple formats"""
        # Generate some test data
        self.logger.info("Test message", component="test")
        
        with self.logger.performance_monitor("export_test"):
            time.sleep(0.001)
        
        # Export logs
        output_files = self.logger.export_logs("all")
        
        # Check that files were created
        assert "json" in output_files
        assert "report" in output_files
        
        # Verify JSON file content
        json_file = Path(output_files["json"])
        assert json_file.exists()
        
        with open(json_file) as f:
            data = json.load(f)
        
        assert "session_metadata" in data
        assert "log_entries" in data
        assert "performance_metrics" in data
        
        # Verify report file
        report_file = Path(output_files["report"])
        assert report_file.exists()
    
    def test_performance_summary_dataframe(self):
        """Test performance metrics DataFrame generation"""
        # Generate test performance data
        with self.logger.performance_monitor("test_op_1", input_size=100):
            time.sleep(0.001)
        
        with self.logger.performance_monitor("test_op_2", input_size=200):
            time.sleep(0.002)
        
        df = self.logger.get_performance_summary()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2
        assert "operation" in df.columns
        assert "duration_ms" in df.columns
        assert "memory_delta_mb" in df.columns
        assert "input_size" in df.columns


class TestDocumentationGenerator:
    """Test documentation generation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MISOLogger(log_dir=self.temp_dir)
        self.doc_gen = DocumentationGenerator(self.logger)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_step_documentation_creation(self):
        """Test creation of step documentation"""
        markdown = self.doc_gen.create_step_documentation(
            step_name="Test Processing Step",
            description="This is a test processing step",
            inputs=["input_data.csv", "config.yaml"],
            outputs=["processed_data.csv", "summary_stats.json"],
            methodology="Apply test transformation using test algorithm",
            requirements=["REQ-1.1", "REQ-2.3"],
            assumptions=["Data is clean", "No missing values"],
            limitations=["Limited to test scenarios", "Requires manual validation"]
        )
        
        assert isinstance(markdown, str)
        assert "Test Processing Step" in markdown
        assert "input_data.csv" in markdown
        assert "processed_data.csv" in markdown
        assert "REQ-1.1" in markdown
        assert "Data is clean" in markdown
        assert "Limited to test scenarios" in markdown
        
        # Check that documentation was logged
        assert len(self.doc_gen.documentation_blocks) == 1
        doc_block = self.doc_gen.documentation_blocks[0]
        assert doc_block["step_name"] == "Test Processing Step"
    
    def test_configuration_documentation(self):
        """Test configuration documentation generation"""
        test_config = {
            'runtime': {
                'mode': 'demo',
                'horizons_h': [12, 24, 36, 48],
                'crs': 'EPSG:4326',
                'random_seed': 42
            },
            'weights': {
                'hazard': {'thermal': 0.3, 'wind': 0.3, 'precip': 0.25},
                'exposure': {'pop': 0.7, 'load': 0.3},
                'vulnerability': {'renew_share': 0.6, 'tx_scarcity': 0.3},
                'blend': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
            },
            'thresholds': {
                'thermal': {'heat_low': 85, 'heat_high': 100},
                'wind': {'gust_low': 20, 'gust_high': 50},
                'precip': {'rain_heavy': 10, 'snow_heavy': 5}
            }
        }
        
        markdown = self.doc_gen.create_configuration_documentation(test_config)
        
        assert isinstance(markdown, str)
        assert "Configuration Parameters" in markdown
        assert "Runtime Configuration" in markdown
        assert "Risk Scoring Weights" in markdown
        assert "Scoring Thresholds" in markdown
        assert "demo" in markdown
        assert "0.3" in markdown  # Check weight values
        assert "85°F" in markdown  # Check threshold values
    
    def test_documentation_export(self):
        """Test export of comprehensive documentation"""
        # Create some test documentation
        self.doc_gen.create_step_documentation(
            step_name="Step 1",
            description="First step",
            inputs=["input1"],
            outputs=["output1"],
            methodology="Method 1",
            requirements=["REQ-1"]
        )
        
        self.doc_gen.create_step_documentation(
            step_name="Step 2", 
            description="Second step",
            inputs=["output1"],
            outputs=["final_output"],
            methodology="Method 2",
            requirements=["REQ-2"]
        )
        
        # Export documentation
        doc_file = self.doc_gen.export_documentation(self.temp_dir)
        
        # Verify file was created
        assert Path(doc_file).exists()
        
        # Check content
        with open(doc_file) as f:
            content = f.read()
        
        assert "MISO Weather-Stress Heatmap - Processing Documentation" in content
        assert "Step 1" in content
        assert "Step 2" in content
        assert "First step" in content
        assert "Second step" in content


class TestGlobalLoggerFunctions:
    """Test global logger utility functions"""
    
    def test_get_logger_singleton(self):
        """Test global logger singleton behavior"""
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return same instance
        assert logger1 is logger2
    
    def test_initialize_logging(self):
        """Test logging initialization function"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            logger = initialize_logging(
                log_dir=temp_dir,
                log_level="WARNING",
                enable_performance=False
            )
            
            assert logger.log_dir == Path(temp_dir)
            assert logger.log_level == 30  # WARNING level
            assert logger.enable_performance is False
            
            logger.close()
        finally:
            shutil.rmtree(temp_dir)


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MISOLogger(log_dir=self.temp_dir)
        self.doc_gen = DocumentationGenerator(self.logger)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_complete_analysis_workflow_logging(self):
        """Test logging for a complete analysis workflow"""
        # Simulate complete workflow
        
        # 1. Configuration
        config = {
            'runtime': {'mode': 'demo', 'seed': 42},
            'weights': {'hazard': {'thermal': 0.3}}
        }
        self.logger.log_configuration(config, "workflow_config")
        
        # 2. Data ingestion
        self.logger.log_data_source(
            "NOAA_API", "https://api.weather.gov", "weather"
        )
        self.logger.log_data_source(
            "EIA_Data", "https://eia.gov/data", "capacity"
        )
        
        # 3. Processing steps with documentation
        with self.logger.performance_monitor("spatial_grid_generation"):
            self.doc_gen.create_step_documentation(
                step_name="Spatial Grid Generation",
                description="Generate hexagonal grid for MISO footprint",
                inputs=["MISO state boundaries"],
                outputs=["hex_grid.geojson"],
                methodology="H3 hexagonal tiling with 40km spacing",
                requirements=["1.2", "7.4"]
            )
            time.sleep(0.01)  # Simulate processing
        
        with self.logger.performance_monitor("weather_processing"):
            self.doc_gen.create_step_documentation(
                step_name="Weather Data Processing",
                description="Process weather forecasts into risk features",
                inputs=["NOAA forecast data", "hex_grid.geojson"],
                outputs=["weather_features.csv"],
                methodology="Spatial aggregation and feature engineering",
                requirements=["2.1", "2.3", "4.1"]
            )
            time.sleep(0.01)
        
        with self.logger.performance_monitor("risk_calculation"):
            self.doc_gen.create_step_documentation(
                step_name="Risk Score Calculation",
                description="Calculate final risk scores",
                inputs=["weather_features.csv", "infrastructure_data.csv"],
                outputs=["risk_scores.csv"],
                methodology="Weighted combination with z-score normalization",
                requirements=["4.4", "5.4"]
            )
            time.sleep(0.01)
        
        # 4. Generate reports
        reproducibility_report = self.logger.generate_reproducibility_report()
        doc_file = self.doc_gen.export_documentation(self.temp_dir)
        log_files = self.logger.export_logs("all")
        
        # Verify comprehensive logging
        assert len(self.logger.log_entries) >= 10
        assert len(self.logger.performance_metrics) == 3
        assert len(self.logger.data_sources) == 2
        assert len(self.doc_gen.documentation_blocks) == 3
        
        # Verify report completeness
        report = reproducibility_report["reproducibility_report"]
        assert report["session_summary"]["log_entries_count"] >= 10
        assert report["performance_summary"]["total_operations"] == 3
        assert len(report["data_sources"]) == 2
        
        # Verify documentation export
        assert Path(doc_file).exists()
        with open(doc_file) as f:
            doc_content = f.read()
        assert "Spatial Grid Generation" in doc_content
        assert "Weather Data Processing" in doc_content
        assert "Risk Score Calculation" in doc_content
        
        # Verify log exports
        assert "json" in log_files
        assert "report" in log_files
        assert Path(log_files["json"]).exists()
        assert Path(log_files["report"]).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])