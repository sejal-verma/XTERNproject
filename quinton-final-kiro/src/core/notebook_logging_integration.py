"""
Notebook Integration for Comprehensive Logging and Documentation

This module provides seamless integration of the logging system into the 
Jupyter notebook environment, including:
- Automatic logging setup and configuration
- Performance monitoring for notebook cells
- Documentation generation for processing steps
- Reproducibility tracking and reporting

Usage in notebook:
```python
from notebook_logging_integration import setup_notebook_logging, log_step, performance_monitor

# Initialize logging
logger, doc_gen = setup_notebook_logging()

# Log processing steps
with log_step("Data Loading", logger, doc_gen):
    # Your processing code here
    pass

# Monitor performance
with performance_monitor("Spatial Processing", logger):
    # Your spatial operations here
    pass
```
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from contextlib import contextmanager
import yaml
import json

# Import our logging system
from logging_system import MISOLogger, DocumentationGenerator, initialize_logging


def setup_notebook_logging(
    log_dir: str = "output/logs",
    log_level: str = "INFO",
    enable_performance: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[MISOLogger, DocumentationGenerator]:
    """
    Setup comprehensive logging for notebook environment
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_performance: Enable performance monitoring
        config: Configuration dictionary to log
        
    Returns:
        Tuple of (logger, documentation_generator)
    """
    # Initialize logger
    logger = initialize_logging(
        log_dir=log_dir,
        log_level=log_level,
        enable_performance=enable_performance
    )
    
    # Initialize documentation generator
    doc_gen = DocumentationGenerator(logger)
    
    # Log notebook initialization
    logger.info("Notebook logging system initialized", 
               component="notebook_integration")
    
    # Log configuration if provided
    if config:
        logger.log_configuration(config, "notebook_config")
        
        # Generate configuration documentation
        config_doc = doc_gen.create_configuration_documentation(config)
        
        # Display configuration documentation in notebook
        try:
            from IPython.display import Markdown, display
            display(Markdown(config_doc))
        except ImportError:
            # Not in notebook environment, just log
            logger.info("Configuration documentation generated", 
                       component="documentation")
    
    # Log system information
    logger.info(f"Python version: {sys.version}", component="system_info")
    logger.info(f"Working directory: {os.getcwd()}", component="system_info")
    
    return logger, doc_gen


@contextmanager
def log_step(step_name: str, 
            logger: MISOLogger,
            doc_gen: DocumentationGenerator,
            description: str = "",
            inputs: Optional[List[str]] = None,
            outputs: Optional[List[str]] = None,
            methodology: str = "",
            requirements: Optional[List[str]] = None,
            assumptions: Optional[List[str]] = None,
            limitations: Optional[List[str]] = None,
            display_docs: bool = True):
    """
    Context manager for logging and documenting processing steps
    
    Args:
        step_name: Name of the processing step
        logger: MISOLogger instance
        doc_gen: DocumentationGenerator instance
        description: Description of the step
        inputs: List of input data/files
        outputs: List of output data/files
        methodology: Methodology description
        requirements: List of requirements addressed
        assumptions: List of key assumptions
        limitations: List of limitations
        display_docs: Whether to display documentation in notebook
    """
    logger.info(f"Starting step: {step_name}", component="processing_step")
    
    # Generate step documentation
    if description or inputs or outputs or methodology:
        step_doc = doc_gen.create_step_documentation(
            step_name=step_name,
            description=description or f"Processing step: {step_name}",
            inputs=inputs or [],
            outputs=outputs or [],
            methodology=methodology or "See code implementation below",
            requirements=requirements or [],
            assumptions=assumptions,
            limitations=limitations
        )
        
        # Display documentation in notebook if requested
        if display_docs:
            try:
                from IPython.display import Markdown, display
                display(Markdown(step_doc))
            except ImportError:
                pass
    
    # Performance monitoring context
    with logger.performance_monitor(f"step_{step_name.lower().replace(' ', '_')}"):
        try:
            yield
            logger.info(f"Completed step: {step_name}", component="processing_step")
        except Exception as e:
            logger.exception(f"Error in step {step_name}: {str(e)}", 
                           component="processing_step")
            raise


@contextmanager 
def performance_monitor(operation_name: str, 
                       logger: MISOLogger,
                       input_size: Optional[int] = None,
                       log_results: bool = True):
    """
    Context manager for performance monitoring with optional result logging
    
    Args:
        operation_name: Name of the operation being monitored
        logger: MISOLogger instance
        input_size: Size of input data (for performance analysis)
        log_results: Whether to log performance results
    """
    with logger.performance_monitor(operation_name, input_size) as monitor:
        yield monitor
        
        if log_results and logger.performance_metrics:
            # Get the latest performance metric
            latest_metric = logger.performance_metrics[-1]
            
            # Log performance summary
            logger.info(
                f"Performance: {operation_name} completed in "
                f"{latest_metric.duration_ms:.1f}ms "
                f"(Memory: {latest_metric.memory_end_mb - latest_metric.memory_start_mb:+.1f}MB)",
                component="performance"
            )


def create_notebook_cell_documentation(
    cell_title: str,
    cell_description: str,
    code_summary: str,
    expected_outputs: List[str],
    doc_gen: DocumentationGenerator,
    display: bool = True
) -> str:
    """
    Create documentation for a notebook cell
    
    Args:
        cell_title: Title of the notebook cell
        cell_description: Description of what the cell does
        code_summary: Summary of the code implementation
        expected_outputs: List of expected outputs
        doc_gen: DocumentationGenerator instance
        display: Whether to display the documentation
        
    Returns:
        Generated markdown documentation
    """
    markdown = f"""
### {cell_title}

**Purpose:** {cell_description}

**Implementation:** {code_summary}

**Expected Outputs:**
{chr(10).join(f'- {output}' for output in expected_outputs)}

**Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
    
    if display:
        try:
            from IPython.display import Markdown, display as ipython_display
            ipython_display(Markdown(markdown))
        except ImportError:
            pass
    
    return markdown


def generate_notebook_summary(
    logger: MISOLogger,
    doc_gen: DocumentationGenerator,
    save_reports: bool = True,
    display_summary: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive summary of notebook execution
    
    Args:
        logger: MISOLogger instance
        doc_gen: DocumentationGenerator instance
        save_reports: Whether to save reports to files
        display_summary: Whether to display summary in notebook
        
    Returns:
        Dictionary containing summary information
    """
    # Generate reproducibility report
    repro_report = logger.generate_reproducibility_report()
    
    # Get performance summary
    perf_df = logger.get_performance_summary()
    
    # Create execution summary
    execution_summary = {
        'session_id': logger.session_id,
        'total_duration_minutes': (datetime.now() - logger.start_time).total_seconds() / 60,
        'total_log_entries': len(logger.log_entries),
        'total_operations': len(logger.performance_metrics),
        'total_data_sources': len(logger.data_sources),
        'error_count': len([e for e in logger.log_entries if e.level == "ERROR"]),
        'warning_count': len([e for e in logger.log_entries if e.level == "WARNING"])
    }
    
    if save_reports:
        # Export all logs and documentation
        log_files = logger.export_logs("all")
        doc_file = doc_gen.export_documentation()
        
        execution_summary['exported_files'] = {
            'logs': log_files,
            'documentation': doc_file
        }
    
    # Display summary in notebook
    if display_summary:
        try:
            from IPython.display import Markdown, display
            
            summary_markdown = f"""
## Notebook Execution Summary

**Session ID:** {execution_summary['session_id']}
**Total Duration:** {execution_summary['total_duration_minutes']:.1f} minutes
**Log Entries:** {execution_summary['total_log_entries']}
**Operations Monitored:** {execution_summary['total_operations']}
**Data Sources Accessed:** {execution_summary['total_data_sources']}
**Errors:** {execution_summary['error_count']}
**Warnings:** {execution_summary['warning_count']}

### Performance Summary
"""
            
            if not perf_df.empty:
                # Add top 5 slowest operations
                top_operations = perf_df.nlargest(5, 'duration_ms')
                summary_markdown += """
**Top 5 Slowest Operations:**
"""
                for _, row in top_operations.iterrows():
                    summary_markdown += f"- {row['operation']}: {row['duration_ms']:.1f}ms\n"
            
            if save_reports and 'exported_files' in execution_summary:
                summary_markdown += f"""
### Exported Files
**Logs:** {len(execution_summary['exported_files']['logs'])} files
**Documentation:** {execution_summary['exported_files']['documentation']}
"""
            
            display(Markdown(summary_markdown))
            
        except ImportError:
            pass
    
    return {
        'execution_summary': execution_summary,
        'reproducibility_report': repro_report,
        'performance_data': perf_df.to_dict('records') if not perf_df.empty else []
    }


def log_data_source_access(
    source_name: str,
    url: str,
    data_type: str,
    logger: MISOLogger,
    coverage_area: Optional[str] = None,
    temporal_range: Optional[str] = None,
    cache_status: str = "fresh",
    error_status: Optional[str] = None
):
    """
    Convenience function for logging data source access from notebook
    
    Args:
        source_name: Name of the data source
        url: URL or path to data source
        data_type: Type of data (weather, infrastructure, etc.)
        logger: MISOLogger instance
        coverage_area: Geographic coverage area
        temporal_range: Temporal coverage range
        cache_status: Cache status (fresh, cached, stale)
        error_status: Error status if any
    """
    logger.log_data_source(
        source_name=source_name,
        url=url,
        data_type=data_type,
        coverage_area=coverage_area,
        temporal_range=temporal_range,
        cache_status=cache_status,
        error_status=error_status
    )
    
    # Display data source info in notebook
    try:
        from IPython.display import Markdown, display
        
        status_emoji = "✅" if not error_status else "❌"
        cache_emoji = "🔄" if cache_status == "cached" else "🆕" if cache_status == "fresh" else "⚠️"
        
        source_info = f"""
**Data Source Accessed:** {status_emoji} {source_name}
- **Type:** {data_type}
- **URL:** {url}
- **Cache Status:** {cache_emoji} {cache_status}
"""
        
        if coverage_area:
            source_info += f"- **Coverage:** {coverage_area}\n"
        if temporal_range:
            source_info += f"- **Temporal Range:** {temporal_range}\n"
        if error_status:
            source_info += f"- **Error:** {error_status}\n"
        
        display(Markdown(source_info))
        
    except ImportError:
        pass


# Notebook magic functions (if IPython is available)
try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    
    @magics_class
    class MISOLoggingMagics(Magics):
        """IPython magic commands for MISO logging system"""
        
        def __init__(self, shell):
            super().__init__(shell)
            self.logger = None
            self.doc_gen = None
        
        @line_magic
        def miso_init_logging(self, line):
            """Initialize MISO logging system"""
            args = line.split()
            log_level = args[0] if args else "INFO"
            
            self.logger, self.doc_gen = setup_notebook_logging(log_level=log_level)
            print(f"✓ MISO logging initialized with level: {log_level}")
            
            # Make logger available in notebook namespace
            self.shell.user_ns['miso_logger'] = self.logger
            self.shell.user_ns['miso_doc_gen'] = self.doc_gen
        
        @cell_magic
        @magic_arguments()
        @argument('--name', type=str, help='Name of the processing step')
        @argument('--desc', type=str, help='Description of the step')
        def miso_step(self, line, cell):
            """Execute cell with automatic step logging and documentation"""
            args = parse_argstring(self.miso_step, line)
            
            if not self.logger:
                print("⚠️ Please run %miso_init_logging first")
                return
            
            step_name = args.name or "Unnamed Step"
            description = args.desc or f"Processing step: {step_name}"
            
            with log_step(step_name, self.logger, self.doc_gen, description=description):
                # Execute the cell code
                self.shell.run_cell(cell)
    
    # Register magics if in IPython environment
    def load_ipython_extension(ipython):
        ipython.register_magic_function(MISOLoggingMagics)
        
except ImportError:
    # Not in IPython environment, skip magic functions
    pass