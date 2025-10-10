"""
Comprehensive Logging and Documentation System for MISO Weather-Stress Heatmap

This module provides comprehensive logging, performance monitoring, and documentation
generation capabilities for the MISO Weather-Stress Heatmap system.

Key Features:
- Structured logging with multiple output formats (console, file, JSON)
- Performance monitoring and timing for spatial operations
- Configuration parameter tracking and validation
- Data source URL and metadata logging
- Reproducibility report generation
- Inline documentation generation for notebook cells

Requirements addressed: 6.4, 7.3
"""

import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
import yaml
import pandas as pd
import geopandas as gpd
import numpy as np
from contextlib import contextmanager
import psutil
import platform
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError


@dataclass
class LogEntry:
    """Structured log entry for comprehensive tracking"""
    timestamp: datetime
    level: str
    component: str
    operation: str
    message: str
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    data_sources: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for spatial operations"""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    cpu_percent: Optional[float] = None


@dataclass
class DataSourceMetadata:
    """Metadata for tracking data sources"""
    source_name: str
    url: str
    access_time: datetime
    data_type: str
    coverage_area: Optional[str] = None
    temporal_range: Optional[str] = None
    api_version: Optional[str] = None
    cache_status: str = "fresh"  # fresh, cached, stale
    error_status: Optional[str] = None


class MISOLogger:
    """Comprehensive logging system for MISO Weather-Stress Heatmap"""
    
    def __init__(self, 
                 log_dir: str = "output/logs",
                 log_level: str = "INFO",
                 enable_performance: bool = True,
                 enable_json_logs: bool = True):
        """
        Initialize comprehensive logging system
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_performance: Enable performance monitoring
            enable_json_logs: Enable structured JSON logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.enable_performance = enable_performance
        self.enable_json_logs = enable_json_logs
        
        # Session metadata (must be set before _setup_logger)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        
        # Initialize logging components
        self.logger = self._setup_logger()
        self.log_entries: List[LogEntry] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.data_sources: List[DataSourceMetadata] = []
        self.configuration_log: Dict[str, Any] = {}
        
        # Performance monitoring
        self.operation_stack: List[Dict] = []
        self.process = psutil.Process()
        
        self.info("MISOLogger initialized", component="logging_system")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup multi-format logger with file and console handlers"""
        logger = logging.getLogger("MISO_WeatherStress")
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler for detailed logs
        log_file = self.log_dir / f"miso_heatmap_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_configuration(self, config: Dict[str, Any], source: str = "config"):
        """Log configuration parameters comprehensively"""
        self.configuration_log[source] = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'source': source
        }
        
        # Log key configuration parameters
        self.info(f"Configuration loaded from {source}", 
                 component="configuration",
                 parameters=config)
        
        # Log specific important parameters
        if 'weights' in config:
            self.info(f"Risk scoring weights configured", 
                     component="configuration",
                     parameters=config['weights'])
        
        if 'runtime' in config:
            self.info(f"Runtime configuration set", 
                     component="configuration",
                     parameters=config['runtime'])
    
    def log_data_source(self, 
                       source_name: str,
                       url: str,
                       data_type: str,
                       coverage_area: Optional[str] = None,
                       temporal_range: Optional[str] = None,
                       api_version: Optional[str] = None,
                       cache_status: str = "fresh",
                       error_status: Optional[str] = None):
        """Log data source access and metadata"""
        metadata = DataSourceMetadata(
            source_name=source_name,
            url=url,
            access_time=datetime.now(),
            data_type=data_type,
            coverage_area=coverage_area,
            temporal_range=temporal_range,
            api_version=api_version,
            cache_status=cache_status,
            error_status=error_status
        )
        
        self.data_sources.append(metadata)
        
        log_msg = f"Data source accessed: {source_name} ({data_type})"
        if error_status:
            self.warning(log_msg + f" - Error: {error_status}",
                        component="data_ingestion",
                        data_sources=[url])
        else:
            self.info(log_msg,
                     component="data_ingestion", 
                     data_sources=[url])
    
    @contextmanager
    def performance_monitor(self, 
                          operation_name: str,
                          input_size: Optional[int] = None):
        """Context manager for performance monitoring of operations"""
        if not self.enable_performance:
            yield
            return
        
        # Start monitoring
        start_time = datetime.now()
        memory_start = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_start = self.process.cpu_percent()
        
        operation_info = {
            'name': operation_name,
            'start_time': start_time,
            'memory_start': memory_start,
            'input_size': input_size
        }
        self.operation_stack.append(operation_info)
        
        self.debug(f"Starting operation: {operation_name}",
                  component="performance")
        
        try:
            yield
        finally:
            # End monitoring
            end_time = datetime.now()
            memory_end = self.process.memory_info().rss / 1024 / 1024  # MB
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get peak memory (approximate)
            memory_peak = max(memory_start, memory_end)
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_start_mb=memory_start,
                memory_end_mb=memory_end,
                memory_peak_mb=memory_peak,
                input_size=input_size,
                cpu_percent=self.process.cpu_percent()
            )
            
            self.performance_metrics.append(metrics)
            self.operation_stack.pop()
            
            # Log performance summary
            self.info(f"Operation completed: {operation_name} "
                     f"({duration_ms:.1f}ms, {memory_end-memory_start:+.1f}MB)",
                     component="performance",
                     parameters={
                         'duration_ms': duration_ms,
                         'memory_delta_mb': memory_end - memory_start,
                         'input_size': input_size
                     })
    
    def timing_decorator(self, operation_name: Optional[str] = None):
        """Decorator for automatic performance monitoring of functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Try to estimate input size from arguments
                input_size = None
                for arg in args:
                    if hasattr(arg, '__len__'):
                        input_size = len(arg)
                        break
                
                with self.performance_monitor(op_name, input_size):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _create_log_entry(self, 
                         level: str,
                         message: str,
                         component: str,
                         operation: Optional[str] = None,
                         duration_ms: Optional[float] = None,
                         data_sources: Optional[List[str]] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         error: Optional[str] = None) -> LogEntry:
        """Create structured log entry"""
        memory_mb = None
        if self.enable_performance:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            operation=operation or "general",
            message=message,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            data_sources=data_sources,
            parameters=parameters,
            error=error
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        entry = self._create_log_entry("DEBUG", message, **kwargs)
        self.log_entries.append(entry)
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self.log_entries.append(entry)
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self.log_entries.append(entry)
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self.log_entries.append(entry)
        self.logger.error(message)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        error_trace = traceback.format_exc()
        kwargs['error'] = error_trace
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self.log_entries.append(entry)
        self.logger.exception(message)
    
    def generate_reproducibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report"""
        # System information
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Package versions
        package_versions = {}
        key_packages = [
            'numpy', 'pandas', 'geopandas', 'shapely', 'folium', 
            'plotly', 'requests', 'scipy', 'scikit-learn', 'h3',
            'pyproj', 'rasterio', 'psutil'
        ]
        
        for package in key_packages:
            try:
                pkg_version = version(package)
                package_versions[package] = pkg_version
            except PackageNotFoundError:
                package_versions[package] = "not_installed"
        
        # Random seeds and configuration
        random_state = {
            'numpy_seed': np.random.get_state()[1][0] if hasattr(np.random, 'get_state') else None,
            'configuration': self.configuration_log
        }
        
        # Data source summary
        data_source_summary = []
        for source in self.data_sources:
            data_source_summary.append({
                'source_name': source.source_name,
                'url': source.url,
                'access_time': source.access_time.isoformat(),
                'data_type': source.data_type,
                'cache_status': source.cache_status,
                'error_status': source.error_status
            })
        
        # Performance summary
        performance_summary = {
            'total_operations': len(self.performance_metrics),
            'total_duration_ms': sum(m.duration_ms for m in self.performance_metrics),
            'peak_memory_mb': max((m.memory_peak_mb for m in self.performance_metrics), default=0),
            'operations': [
                {
                    'name': m.operation_name,
                    'duration_ms': m.duration_ms,
                    'memory_delta_mb': m.memory_end_mb - m.memory_start_mb,
                    'input_size': m.input_size
                }
                for m in self.performance_metrics
            ]
        }
        
        # Session summary
        session_summary = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'log_entries_count': len(self.log_entries),
            'error_count': len([e for e in self.log_entries if e.level == "ERROR"]),
            'warning_count': len([e for e in self.log_entries if e.level == "WARNING"])
        }
        
        return {
            'reproducibility_report': {
                'generated_at': datetime.now().isoformat(),
                'system_info': system_info,
                'package_versions': package_versions,
                'random_state': random_state,
                'data_sources': data_source_summary,
                'performance_summary': performance_summary,
                'session_summary': session_summary
            }
        }
    
    def export_logs(self, format_type: str = "all") -> Dict[str, str]:
        """Export logs in various formats"""
        output_files = {}
        
        if format_type in ["all", "json"]:
            # JSON structured logs
            json_file = self.log_dir / f"structured_logs_{self.session_id}.json"
            structured_data = {
                'session_metadata': {
                    'session_id': self.session_id,
                    'start_time': self.start_time.isoformat(),
                    'export_time': datetime.now().isoformat()
                },
                'log_entries': [asdict(entry) for entry in self.log_entries],
                'performance_metrics': [asdict(metric) for metric in self.performance_metrics],
                'data_sources': [asdict(source) for source in self.data_sources],
                'configuration': self.configuration_log
            }
            
            with open(json_file, 'w') as f:
                json.dump(structured_data, f, indent=2, default=str)
            output_files['json'] = str(json_file)
        
        if format_type in ["all", "csv"]:
            # CSV performance metrics
            if self.performance_metrics:
                csv_file = self.log_dir / f"performance_metrics_{self.session_id}.csv"
                df = pd.DataFrame([asdict(metric) for metric in self.performance_metrics])
                df.to_csv(csv_file, index=False)
                output_files['csv'] = str(csv_file)
        
        if format_type in ["all", "report"]:
            # Reproducibility report
            report_file = self.log_dir / f"reproducibility_report_{self.session_id}.json"
            report = self.generate_reproducibility_report()
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            output_files['report'] = str(report_file)
        
        return output_files
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance metrics as DataFrame for analysis"""
        if not self.performance_metrics:
            return pd.DataFrame()
        
        data = []
        for metric in self.performance_metrics:
            data.append({
                'operation': metric.operation_name,
                'duration_ms': metric.duration_ms,
                'memory_start_mb': metric.memory_start_mb,
                'memory_end_mb': metric.memory_end_mb,
                'memory_delta_mb': metric.memory_end_mb - metric.memory_start_mb,
                'input_size': metric.input_size,
                'cpu_percent': metric.cpu_percent,
                'timestamp': metric.start_time
            })
        
        return pd.DataFrame(data)
    
    def close(self):
        """Close logger and export final logs"""
        self.info("Closing MISOLogger and exporting final logs", 
                 component="logging_system")
        
        # Export all log formats
        output_files = self.export_logs("all")
        
        # Log export summary
        for format_type, file_path in output_files.items():
            self.info(f"Exported {format_type} logs to: {file_path}",
                     component="logging_system")
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
        
        return output_files


class DocumentationGenerator:
    """Generate inline documentation for notebook cells and processing steps"""
    
    def __init__(self, logger: MISOLogger):
        self.logger = logger
        self.documentation_blocks: List[Dict[str, Any]] = []
    
    def create_step_documentation(self, 
                                step_name: str,
                                description: str,
                                inputs: List[str],
                                outputs: List[str],
                                methodology: str,
                                requirements: List[str],
                                assumptions: Optional[List[str]] = None,
                                limitations: Optional[List[str]] = None) -> str:
        """Create comprehensive documentation for a processing step"""
        
        doc_block = {
            'step_name': step_name,
            'description': description,
            'inputs': inputs,
            'outputs': outputs,
            'methodology': methodology,
            'requirements': requirements,
            'assumptions': assumptions or [],
            'limitations': limitations or [],
            'timestamp': datetime.now().isoformat()
        }
        
        self.documentation_blocks.append(doc_block)
        
        # Generate markdown documentation
        markdown = f"""
## {step_name}

**Description:** {description}

**Inputs:**
{chr(10).join(f'- {inp}' for inp in inputs)}

**Outputs:**
{chr(10).join(f'- {out}' for out in outputs)}

**Methodology:**
{methodology}

**Requirements Addressed:**
{chr(10).join(f'- {req}' for req in requirements)}
"""
        
        if assumptions:
            markdown += f"""
**Key Assumptions:**
{chr(10).join(f'- {assumption}' for assumption in assumptions)}
"""
        
        if limitations:
            markdown += f"""
**Limitations:**
{chr(10).join(f'- {limitation}' for limitation in limitations)}
"""
        
        markdown += f"""
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
        
        self.logger.info(f"Generated documentation for step: {step_name}",
                        component="documentation")
        
        return markdown
    
    def create_configuration_documentation(self, config: Dict[str, Any]) -> str:
        """Create documentation for configuration parameters"""
        
        markdown = """
## Configuration Parameters

This section documents all configuration parameters used in the analysis.

### Runtime Configuration
"""
        
        if 'runtime' in config:
            runtime = config['runtime']
            markdown += f"""
- **Mode:** {runtime.get('mode', 'not_specified')}
- **Forecast Horizons:** {runtime.get('horizons_h', 'not_specified')}
- **Coordinate System:** {runtime.get('crs', 'not_specified')}
- **Random Seed:** {runtime.get('random_seed', 'not_specified')}
"""
        
        if 'weights' in config:
            weights = config['weights']
            markdown += """
### Risk Scoring Weights

#### Hazard Component Weights
"""
            if 'hazard' in weights:
                hazard = weights['hazard']
                for component, weight in hazard.items():
                    markdown += f"- **{component.title()}:** {weight}\n"
            
            markdown += """
#### Exposure Component Weights
"""
            if 'exposure' in weights:
                exposure = weights['exposure']
                for component, weight in exposure.items():
                    markdown += f"- **{component.title()}:** {weight}\n"
            
            markdown += """
#### Vulnerability Component Weights
"""
            if 'vulnerability' in weights:
                vulnerability = weights['vulnerability']
                for component, weight in vulnerability.items():
                    markdown += f"- **{component.title()}:** {weight}\n"
            
            markdown += """
#### Final Blend Weights
"""
            if 'blend' in weights:
                blend = weights['blend']
                for component, weight in blend.items():
                    markdown += f"- **{component.title()}:** {weight}\n"
        
        if 'thresholds' in config:
            thresholds = config['thresholds']
            markdown += """
### Scoring Thresholds

#### Thermal Stress Thresholds
"""
            if 'thermal' in thresholds:
                thermal = thresholds['thermal']
                for param, value in thermal.items():
                    markdown += f"- **{param.replace('_', ' ').title()}:** {value}°F\n"
            
            markdown += """
#### Wind Stress Thresholds
"""
            if 'wind' in thresholds:
                wind = thresholds['wind']
                for param, value in wind.items():
                    markdown += f"- **{param.replace('_', ' ').title()}:** {value} mph\n"
            
            markdown += """
#### Precipitation Stress Thresholds
"""
            if 'precip' in thresholds:
                precip = thresholds['precip']
                for param, value in precip.items():
                    unit = "mm/h" if "rain" in param else "cm/h" if "snow" in param else ""
                    markdown += f"- **{param.replace('_', ' ').title()}:** {value} {unit}\n"
        
        markdown += f"""
**Configuration documented at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
        
        return markdown
    
    def export_documentation(self, output_dir: str = "output/docs") -> str:
        """Export all documentation to markdown file"""
        doc_dir = Path(output_dir)
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        doc_file = doc_dir / f"processing_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Compile all documentation
        full_doc = f"""# MISO Weather-Stress Heatmap - Processing Documentation

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document provides comprehensive documentation of all processing steps, 
methodologies, and configuration parameters used in the MISO Weather-Stress 
Heatmap analysis.

---

"""
        
        # Add all step documentation
        for doc_block in self.documentation_blocks:
            full_doc += f"""
## {doc_block['step_name']}

**Description:** {doc_block['description']}

**Inputs:**
{chr(10).join(f'- {inp}' for inp in doc_block['inputs'])}

**Outputs:**
{chr(10).join(f'- {out}' for out in doc_block['outputs'])}

**Methodology:**
{doc_block['methodology']}

**Requirements Addressed:**
{chr(10).join(f'- {req}' for req in doc_block['requirements'])}
"""
            
            if doc_block['assumptions']:
                full_doc += f"""
**Key Assumptions:**
{chr(10).join(f'- {assumption}' for assumption in doc_block['assumptions'])}
"""
            
            if doc_block['limitations']:
                full_doc += f"""
**Limitations:**
{chr(10).join(f'- {limitation}' for limitation in doc_block['limitations'])}
"""
            
            full_doc += f"""
**Documented:** {doc_block['timestamp']}

---

"""
        
        # Write to file
        with open(doc_file, 'w') as f:
            f.write(full_doc)
        
        self.logger.info(f"Exported comprehensive documentation to: {doc_file}",
                        component="documentation")
        
        return str(doc_file)


# Global logger instance (initialized when needed)
_global_logger: Optional[MISOLogger] = None

def get_logger() -> MISOLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MISOLogger()
    return _global_logger

def initialize_logging(log_dir: str = "output/logs",
                      log_level: str = "INFO",
                      enable_performance: bool = True) -> MISOLogger:
    """Initialize global logging system"""
    global _global_logger
    _global_logger = MISOLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_performance=enable_performance
    )
    return _global_logger