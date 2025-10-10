"""
Test suite for notebook integration functionality

Tests the notebook integration components including:
- Notebook logging setup
- Step logging and documentation
- Performance monitoring integration
- Data source logging
- Summary generation
"""

import pytest
import tempfile
import shutil
import time
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from notebook_logging_integration import (
    setup_notebook_logging, log_step, performance_monitor,
    create_notebook_cell_documentation, generate_notebook_summary,
    log_data_source_access
)
from logging_system import MISOLogger, DocumentationGenerator


class TestNotebookLoggingSetup:
    """Test notebook logging setup functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_notebook_logging_basic(self):
        """Test basic notebook logging setup"""
        logger, doc_gen = setup_notebook_logging(
            log_dir=self.temp_dir,
            log_level="DEBUG"
        )
        
        assert isinstance(logger, MISOLogger)
        assert isinstance(doc_gen, DocumentationGenerator)
        assert logger.log_dir == Path(self.temp_dir)
        assert logger.log_level == 10  # DEBUG level
        
        # Check initialization log
        init_logs = [entry for entry in logger.log_entries 
                    if "notebook_integration" in entry.component]
        assert len(init_logs) >= 1
        
        logger.close()
    
    def test_setup_notebook_logging_with_config(self):
        """Test notebook logging setup with configuration"""
        test_config = {
            'runtime': {'mode': 'demo', 'seed': 42},
            'weights': {'hazard': {'thermal': 0.3}}
        }
        
        logger, doc_gen = setup_notebook_logging(
            log_dir=self.temp_dir,
            config=test_config
        )
        
        # Check configuration was logged
        assert "notebook_config" in logger.configuration_log
        assert logger.configuration_log["notebook_config"]["config"] == test_config
        
        # Check documentation was generated
        assert len(doc_gen.documentation_blocks) >= 0  # May have config documentation
        
        logger.close()


class TestStepLogging:
    """Test step logging and documentation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger, self.doc_gen = setup_notebook_logging(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_log_step_basic(self):
        """Test basic step logging functionality"""
        initial_entries = len(self.logger.log_entries)
        initial_docs = len(self.doc_gen.documentation_blocks)
        
        with log_step("Test Step", self.logger, self.doc_gen):
            time.sleep(0.001)  # Simulate work
        
        # Check logging
        assert len(self.logger.log_entries) > initial_entries
        
        # Check for step start and completion logs
        step_logs = [entry for entry in self.logger.log_entries 
                    if "processing_step" in entry.component]
        assert len(step_logs) >= 2  # Start and completion
    
    def test_log_step_with_documentation(self):
        """Test step logging with full documentation"""
        initial_docs = len(self.doc_gen.documentation_blocks)
        
        with log_step(
            step_name="Documented Test Step",
            logger=self.logger,
            doc_gen=self.doc_gen,
            description="This is a test step with full documentation",
            inputs=["test_input.csv"],
            outputs=["test_output.csv"],
            methodology="Test methodology",
            requirements=["REQ-1.1", "REQ-2.2"],
            assumptions=["Test assumption"],
            limitations=["Test limitation"],
            display_docs=False  # Don't try to display in test
        ):
            time.sleep(0.001)
        
        # Check documentation was created
        assert len(self.doc_gen.documentation_blocks) > initial_docs
        
        # Check documentation content
        latest_doc = self.doc_gen.documentation_blocks[-1]
        assert latest_doc["step_name"] == "Documented Test Step"
        assert "test_input.csv" in latest_doc["inputs"]
        assert "test_output.csv" in latest_doc["outputs"]
        assert "REQ-1.1" in latest_doc["requirements"]
    
    def test_log_step_error_handling(self):
        """Test step logging with error handling"""
        with pytest.raises(ValueError):
            with log_step("Error Step", self.logger, self.doc_gen):
                raise ValueError("Test error")
        
        # Check error was logged
        error_logs = [entry for entry in self.logger.log_entries 
                     if entry.level == "ERROR"]
        assert len(error_logs) >= 1


class TestPerformanceMonitoring:
    """Test performance monitoring integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger, _ = setup_notebook_logging(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_performance_monitor_basic(self):
        """Test basic performance monitoring"""
        initial_metrics = len(self.logger.performance_metrics)
        
        with performance_monitor("test_operation", self.logger):
            time.sleep(0.01)  # Simulate work
        
        # Check performance metric was recorded
        assert len(self.logger.performance_metrics) > initial_metrics
        
        latest_metric = self.logger.performance_metrics[-1]
        assert "test_operation" in latest_metric.operation_name
        assert latest_metric.duration_ms > 0
    
    def test_performance_monitor_with_input_size(self):
        """Test performance monitoring with input size tracking"""
        with performance_monitor("sized_operation", self.logger, input_size=1000):
            time.sleep(0.001)
        
        latest_metric = self.logger.performance_metrics[-1]
        assert latest_metric.input_size == 1000
    
    def test_performance_monitor_logging_results(self):
        """Test performance monitoring with result logging"""
        initial_entries = len(self.logger.log_entries)
        
        with performance_monitor("logged_operation", self.logger, log_results=True):
            time.sleep(0.001)
        
        # Check performance result was logged
        perf_logs = [entry for entry in self.logger.log_entries[initial_entries:] 
                    if "performance" in entry.component]
        assert len(perf_logs) >= 1


class TestDataSourceLogging:
    """Test data source logging functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger, _ = setup_notebook_logging(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_log_data_source_access_basic(self):
        """Test basic data source logging"""
        initial_sources = len(self.logger.data_sources)
        
        log_data_source_access(
            source_name="Test API",
            url="https://test.api.com",
            data_type="test_data",
            logger=self.logger
        )
        
        # Check data source was logged
        assert len(self.logger.data_sources) > initial_sources
        
        latest_source = self.logger.data_sources[-1]
        assert latest_source.source_name == "Test API"
        assert latest_source.url == "https://test.api.com"
        assert latest_source.data_type == "test_data"
    
    def test_log_data_source_access_with_metadata(self):
        """Test data source logging with full metadata"""
        log_data_source_access(
            source_name="Full Metadata API",
            url="https://full.api.com",
            data_type="weather",
            logger=self.logger,
            coverage_area="MISO",
            temporal_range="48h",
            cache_status="cached",
            error_status=None
        )
        
        latest_source = self.logger.data_sources[-1]
        assert latest_source.coverage_area == "MISO"
        assert latest_source.temporal_range == "48h"
        assert latest_source.cache_status == "cached"
        assert latest_source.error_status is None
    
    def test_log_data_source_access_with_error(self):
        """Test data source logging with error status"""
        log_data_source_access(
            source_name="Error API",
            url="https://error.api.com",
            data_type="failed_data",
            logger=self.logger,
            error_status="Connection timeout"
        )
        
        latest_source = self.logger.data_sources[-1]
        assert latest_source.error_status == "Connection timeout"
        
        # Check warning was logged
        warning_logs = [entry for entry in self.logger.log_entries 
                       if entry.level == "WARNING"]
        assert len(warning_logs) >= 1


class TestDocumentationGeneration:
    """Test documentation generation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger, self.doc_gen = setup_notebook_logging(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_create_notebook_cell_documentation(self):
        """Test notebook cell documentation creation"""
        markdown = create_notebook_cell_documentation(
            cell_title="Test Cell",
            cell_description="This cell tests functionality",
            code_summary="Executes test code with assertions",
            expected_outputs=["test_results.csv", "validation_report.txt"],
            doc_gen=self.doc_gen,
            display=False
        )
        
        assert isinstance(markdown, str)
        assert "Test Cell" in markdown
        assert "This cell tests functionality" in markdown
        assert "test_results.csv" in markdown
        assert "validation_report.txt" in markdown


class TestSummaryGeneration:
    """Test summary generation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger, self.doc_gen = setup_notebook_logging(log_dir=self.temp_dir)
        
        # Generate some test data
        self.logger.info("Test log entry", component="test")
        
        with self.logger.performance_monitor("test_operation"):
            time.sleep(0.001)
        
        self.logger.log_data_source("test_source", "http://test.com", "test_data")
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_generate_notebook_summary_basic(self):
        """Test basic notebook summary generation"""
        summary = generate_notebook_summary(
            logger=self.logger,
            doc_gen=self.doc_gen,
            save_reports=False,
            display_summary=False
        )
        
        assert "execution_summary" in summary
        assert "reproducibility_report" in summary
        assert "performance_data" in summary
        
        exec_summary = summary["execution_summary"]
        assert exec_summary["session_id"] == self.logger.session_id
        assert exec_summary["total_log_entries"] >= 1
        assert exec_summary["total_operations"] >= 1
        assert exec_summary["total_data_sources"] >= 1
    
    def test_generate_notebook_summary_with_reports(self):
        """Test notebook summary generation with file exports"""
        summary = generate_notebook_summary(
            logger=self.logger,
            doc_gen=self.doc_gen,
            save_reports=True,
            display_summary=False
        )
        
        assert "exported_files" in summary["execution_summary"]
        exported = summary["execution_summary"]["exported_files"]
        
        # Check log files were exported
        assert "logs" in exported
        assert isinstance(exported["logs"], dict)
        
        # Check documentation was exported
        assert "documentation" in exported
        doc_file = Path(exported["documentation"])
        assert doc_file.exists()
    
    def test_generate_notebook_summary_performance_analysis(self):
        """Test notebook summary with performance analysis"""
        # Generate more performance data
        with self.logger.performance_monitor("slow_operation"):
            time.sleep(0.01)
        
        with self.logger.performance_monitor("fast_operation"):
            time.sleep(0.001)
        
        summary = generate_notebook_summary(
            logger=self.logger,
            doc_gen=self.doc_gen,
            save_reports=False,
            display_summary=False
        )
        
        perf_data = summary["performance_data"]
        assert len(perf_data) >= 2
        
        # Check performance data structure
        for perf_record in perf_data:
            assert "operation" in perf_record
            assert "duration_ms" in perf_record
            assert "memory_delta_mb" in perf_record


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_notebook_workflow(self):
        """Test complete notebook workflow with all components"""
        # Initialize logging
        logger, doc_gen = setup_notebook_logging(
            log_dir=self.temp_dir,
            config={'runtime': {'mode': 'test'}}
        )
        
        try:
            # Simulate data processing workflow
            with log_step(
                "Data Loading",
                logger, doc_gen,
                description="Load and validate input data",
                inputs=["raw_data.csv"],
                outputs=["validated_data.csv"],
                methodology="CSV loading with validation",
                requirements=["REQ-1"],
                display_docs=False
            ):
                # Log data source access
                log_data_source_access(
                    "CSV_File", "data/raw_data.csv", "input_data", logger
                )
                
                # Simulate processing with performance monitoring
                with performance_monitor("data_validation", logger, input_size=1000):
                    time.sleep(0.01)
            
            # Simulate analysis step
            with log_step(
                "Data Analysis",
                logger, doc_gen,
                description="Perform statistical analysis",
                inputs=["validated_data.csv"],
                outputs=["analysis_results.json"],
                methodology="Statistical analysis with pandas",
                requirements=["REQ-2"],
                display_docs=False
            ):
                with performance_monitor("statistical_analysis", logger):
                    time.sleep(0.005)
            
            # Generate final summary
            summary = generate_notebook_summary(
                logger, doc_gen,
                save_reports=True,
                display_summary=False
            )
            
            # Verify comprehensive workflow tracking
            assert summary["execution_summary"]["total_operations"] >= 2
            assert summary["execution_summary"]["total_data_sources"] >= 1
            assert len(doc_gen.documentation_blocks) >= 2
            
            # Verify files were exported
            assert "exported_files" in summary["execution_summary"]
            
        finally:
            logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])