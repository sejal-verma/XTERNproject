# MISO Weather-Stress Heatmap - Comprehensive Logging and Documentation System

## Overview

This document provides comprehensive guidance on using the logging and documentation system implemented for the MISO Weather-Stress Heatmap project. The system addresses requirements 6.4 and 7.3 by providing:

- **Comprehensive logging** of all configuration parameters and data source URLs
- **Performance monitoring** and timing for large spatial operations
- **Inline documentation** generation explaining each processing step
- **Reproducibility reporting** with random seeds, versions, and data timestamps

## System Components

### 1. Core Logging System (`logging_system.py`)

The `MISOLogger` class provides structured logging with multiple output formats:

```python
from logging_system import MISOLogger, DocumentationGenerator

# Initialize logger
logger = MISOLogger(
    log_dir="output/logs",
    log_level="INFO",
    enable_performance=True
)

# Basic logging
logger.info("Processing started", component="data_processing")
logger.warning("API rate limit approaching", component="weather_ingestion")
logger.error("Failed to load data", component="data_loading")

# Log configuration parameters
config = {
    'runtime': {'mode': 'demo', 'seed': 42},
    'weights': {'hazard': {'thermal': 0.3}}
}
logger.log_configuration(config, "main_config")

# Log data source access
logger.log_data_source(
    source_name="NOAA_API",
    url="https://api.weather.gov/gridpoints",
    data_type="weather_forecast",
    coverage_area="MISO footprint",
    temporal_range="48h"
)
```

### 2. Performance Monitoring

Monitor execution time and memory usage for operations:

```python
# Context manager approach
with logger.performance_monitor("spatial_grid_generation", input_size=1000):
    # Your spatial processing code here
    grid = generate_hex_grid(footprint)

# Decorator approach
@logger.timing_decorator("data_processing")
def process_weather_data(data):
    # Your processing code here
    return processed_data
```

### 3. Documentation Generation

Generate comprehensive documentation for processing steps:

```python
doc_gen = DocumentationGenerator(logger)

# Create step documentation
step_doc = doc_gen.create_step_documentation(
    step_name="Weather Data Ingestion",
    description="Fetch and process weather forecast data",
    inputs=["Grid cells", "API endpoints"],
    outputs=["weather_features.csv"],
    methodology="NOAA/NWS primary with Open-Meteo fallback",
    requirements=["2.1", "2.3", "2.4"],
    assumptions=["API services available"],
    limitations=["48h forecast horizon limit"]
)

# Generate configuration documentation
config_doc = doc_gen.create_configuration_documentation(config)
```

### 4. Notebook Integration (`notebook_logging_integration.py`)

Seamless integration with Jupyter notebooks:

```python
from notebook_logging_integration import (
    setup_notebook_logging, log_step, performance_monitor
)

# Initialize logging in notebook
logger, doc_gen = setup_notebook_logging(
    log_level="INFO",
    config=your_config_dict
)

# Log processing steps with automatic documentation
with log_step(
    step_name="Spatial Grid Generation",
    logger=logger,
    doc_gen=doc_gen,
    description="Generate hexagonal grid for analysis",
    inputs=["MISO boundaries"],
    outputs=["hex_grid.geojson"],
    methodology="H3 hexagonal tiling with 40km spacing",
    requirements=["1.2", "7.4"]
):
    # Your processing code here
    grid = create_miso_grid()

# Performance monitoring
with performance_monitor("weather_processing", logger, input_size=len(grid)):
    weather_data = fetch_weather_data(grid)
```

## Key Features

### 1. Structured Logging

All log entries include:
- Timestamp
- Log level (DEBUG, INFO, WARNING, ERROR)
- Component identifier
- Operation name
- Message content
- Performance metrics (duration, memory usage)
- Data source information
- Configuration parameters

### 2. Performance Monitoring

Automatic tracking of:
- Execution duration (milliseconds)
- Memory usage (start, end, delta)
- Input data size
- CPU utilization
- Operation categorization

### 3. Data Source Tracking

Comprehensive logging of:
- Source name and type
- Access URLs
- Temporal and spatial coverage
- API versions
- Cache status
- Error conditions

### 4. Configuration Management

Complete tracking of:
- Runtime parameters
- Risk scoring weights
- Threshold values
- Random seeds
- Coordinate systems

### 5. Reproducibility Reporting

Automatic generation of:
- System information (platform, Python version)
- Package versions
- Random state
- Data source metadata
- Performance summaries
- Session statistics

## Usage Examples

### Basic Notebook Integration

```python
# Cell 1: Initialize logging
from notebook_logging_integration import setup_notebook_logging

logger, doc_gen = setup_notebook_logging(
    config={
        'runtime': {'mode': 'demo', 'seed': 42},
        'weights': {'hazard': {'thermal': 0.3, 'wind': 0.3}}
    }
)

# Cell 2: Process data with logging
with log_step("Data Loading", logger, doc_gen,
              description="Load and validate input data",
              inputs=["raw_data.csv"],
              outputs=["clean_data.csv"]):
    
    # Log data source access
    log_data_source_access(
        "CSV_File", "data/raw_data.csv", "input_data", logger
    )
    
    # Process with performance monitoring
    with performance_monitor("data_validation", logger):
        data = pd.read_csv("data/raw_data.csv")
        clean_data = validate_data(data)

# Cell 3: Generate final reports
summary = generate_notebook_summary(logger, doc_gen, save_reports=True)
```

### Advanced Performance Analysis

```python
# Get performance summary as DataFrame
perf_df = logger.get_performance_summary()

# Analyze slowest operations
top_slow = perf_df.nlargest(5, 'duration_ms')
print("Slowest Operations:")
for _, row in top_slow.iterrows():
    print(f"  {row['operation']}: {row['duration_ms']:.1f}ms")

# Memory usage analysis
memory_intensive = perf_df[perf_df['memory_delta_mb'] > 100]
print(f"Memory-intensive operations: {len(memory_intensive)}")

# Operation categorization
spatial_ops = perf_df[perf_df['operation'].str.contains('spatial|grid')]
weather_ops = perf_df[perf_df['operation'].str.contains('weather|api')]
```

### Custom Documentation Generation

```python
# Create custom cell documentation
cell_doc = create_notebook_cell_documentation(
    cell_title="Risk Score Calculation",
    cell_description="Calculate final risk scores using weighted combination",
    code_summary="Apply formula: Risk = zscore(α×H + β×E + γ×V)",
    expected_outputs=["risk_scores.csv", "confidence_metrics.csv"],
    doc_gen=doc_gen
)

# Export comprehensive documentation
doc_file = doc_gen.export_documentation("output/docs")
print(f"Documentation exported to: {doc_file}")
```

## Output Files

The logging system generates several output files:

### 1. Log Files
- `miso_heatmap_YYYYMMDD_HHMMSS.log` - Detailed text logs
- `structured_logs_YYYYMMDD_HHMMSS.json` - Structured JSON logs
- `performance_metrics_YYYYMMDD_HHMMSS.csv` - Performance data

### 2. Reports
- `reproducibility_report_YYYYMMDD_HHMMSS.json` - Complete reproducibility report
- `processing_documentation_YYYYMMDD_HHMMSS.md` - Step-by-step documentation

### 3. Analysis Files
- Performance summaries and visualizations
- Data source access logs
- Configuration parameter tracking

## Best Practices

### 1. Initialization
- Always initialize logging at the start of your notebook
- Provide configuration parameters for complete tracking
- Use appropriate log levels (DEBUG for development, INFO for production)

### 2. Step Documentation
- Document each major processing step
- Include clear inputs, outputs, and methodology
- Reference specific requirements being addressed
- Note key assumptions and limitations

### 3. Performance Monitoring
- Monitor all spatial operations (typically slow)
- Track API calls and data loading operations
- Include input size information when available
- Use descriptive operation names

### 4. Data Source Logging
- Log every external data access
- Include temporal and spatial coverage information
- Track cache status and error conditions
- Document API versions and endpoints

### 5. Error Handling
- Use appropriate log levels for different conditions
- Include context information in error messages
- Log exceptions with full stack traces
- Document recovery strategies

## Integration with Existing Code

The logging system is designed to integrate seamlessly with existing code:

```python
# Minimal integration - just add logging
from logging_system import get_logger

logger = get_logger()

def existing_function(data):
    logger.info("Processing data", component="existing_module")
    
    with logger.performance_monitor("existing_function"):
        # Your existing code here
        result = process_data(data)
    
    logger.info(f"Processed {len(result)} records", component="existing_module")
    return result

# Enhanced integration - add documentation
from notebook_logging_integration import log_step

with log_step("Data Processing", logger, doc_gen,
              description="Process input data using existing function"):
    result = existing_function(input_data)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Errors**: Check write permissions for log directory
3. **Memory Issues**: Disable performance monitoring for very large operations
4. **Display Issues**: IPython display functions only work in notebook environment

### Performance Considerations

- Logging overhead is minimal (<1% for most operations)
- Performance monitoring adds ~2-5ms per operation
- JSON log files can become large with extensive logging
- Use appropriate log levels to control verbosity

### Configuration Tips

- Use DEBUG level only during development
- Set `enable_performance=False` for production if overhead is a concern
- Customize log directory to avoid conflicts
- Regular cleanup of old log files recommended

## Requirements Compliance

This logging system addresses the following requirements:

### Requirement 6.4: Comprehensive Logging
- ✅ All configuration parameters logged with timestamps
- ✅ Data source URLs and metadata tracked
- ✅ Performance metrics for all operations
- ✅ Error conditions and recovery actions logged

### Requirement 7.3: Documentation and Reproducibility
- ✅ Inline documentation for each processing step
- ✅ Methodology and assumptions documented
- ✅ Random seeds and system information tracked
- ✅ Package versions and dependencies recorded
- ✅ Complete reproducibility reports generated

## Conclusion

The comprehensive logging and documentation system provides complete transparency and reproducibility for the MISO Weather-Stress Heatmap analysis. By following the patterns and examples in this documentation, users can ensure their analysis is fully logged, documented, and reproducible.

For additional examples and advanced usage patterns, see:
- `example_notebook_integration.py` - Complete notebook examples
- `test_logging_system.py` - Unit tests demonstrating functionality
- `test_notebook_integration.py` - Integration test examples