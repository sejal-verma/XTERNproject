"""
Example Notebook Integration Code

This file contains example code that can be copied into notebook cells
to demonstrate comprehensive logging and documentation integration.

Copy these code blocks into your Jupyter notebook cells to enable
comprehensive logging, performance monitoring, and documentation generation.
"""

# =============================================================================
# CELL 1: Initialize Logging System
# =============================================================================

# Import logging system
from logging_system import MISOLogger, DocumentationGenerator
from notebook_logging_integration import (
    setup_notebook_logging, log_step, performance_monitor,
    log_data_source_access, generate_notebook_summary
)

# Initialize comprehensive logging
logger, doc_gen = setup_notebook_logging(
    log_level="INFO",
    enable_performance=True,
    config={
        'runtime': {
            'mode': 'demo',
            'horizons_h': [12, 24, 36, 48],
            'crs': 'EPSG:4326',
            'random_seed': 42
        },
        'weights': {
            'hazard': {'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15},
            'exposure': {'pop': 0.7, 'load': 0.3},
            'vulnerability': {'renew_share': 0.6, 'tx_scarcity': 0.3, 'outage': 0.1},
            'blend': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
        }
    }
)

print("✓ Comprehensive logging system initialized")
print(f"Session ID: {logger.session_id}")

# =============================================================================
# CELL 2: Spatial Grid Generation with Logging
# =============================================================================

with log_step(
    step_name="Spatial Grid Generation",
    logger=logger,
    doc_gen=doc_gen,
    description="Generate hexagonal grid for MISO footprint analysis",
    inputs=["MISO state boundaries", "Configuration parameters"],
    outputs=["hex_grid.geojson", "Grid metadata"],
    methodology="Use H3 hexagonal tiling system with 40km spacing, clipped to MISO footprint",
    requirements=["1.2", "7.4"],
    assumptions=["MISO footprint accurately represented by state boundaries"],
    limitations=["Grid resolution fixed at 40km", "Partial state coverage approximated"]
):
    
    # Import spatial framework with performance monitoring
    with performance_monitor("spatial_imports", logger):
        from spatial_framework import MISOFootprint, HexGridGenerator
    
    # Create MISO footprint
    with performance_monitor("miso_footprint_creation", logger, input_size=50):
        footprint_manager = MISOFootprint(crs="EPSG:4326")
        miso_footprint = footprint_manager.create_miso_footprint(use_simplified=True)
        
        logger.info(f"MISO footprint created: {len(miso_footprint)} polygons, "
                   f"Area: {footprint_manager.area_km2:.0f} km²",
                   component="spatial_processing")
    
    # Generate hexagonal grid
    with performance_monitor("hex_grid_generation", logger, input_size=len(miso_footprint)):
        grid_generator = HexGridGenerator(hex_size_km=40, crs="EPSG:4326")
        hex_grid = grid_generator.generate_grid(miso_footprint)
        
        logger.info(f"Hexagonal grid generated: {len(hex_grid)} cells",
                   component="spatial_processing")
    
    # Save grid data
    output_path = "data/processed/miso_hex_grid.geojson"
    hex_grid.to_file(output_path, driver="GeoJSON")
    
    logger.info(f"Grid saved to: {output_path}", component="data_export")

# =============================================================================
# CELL 3: Weather Data Ingestion with Comprehensive Logging
# =============================================================================

with log_step(
    step_name="Weather Data Ingestion",
    logger=logger,
    doc_gen=doc_gen,
    description="Ingest weather forecast data from NOAA/NWS and Open-Meteo APIs",
    inputs=["Hex grid cells", "API endpoints", "Forecast horizons"],
    outputs=["weather_features.csv", "API response cache"],
    methodology="Primary NOAA/NWS API with Open-Meteo fallback, spatial aggregation to grid cells",
    requirements=["2.1", "2.3", "2.4", "2.5"],
    assumptions=["API services available", "Forecast accuracy within acceptable bounds"],
    limitations=["Dependent on external APIs", "Limited to 48h forecast horizon"]
):
    
    # Import weather adapters
    with performance_monitor("weather_adapter_imports", logger):
        from weather_adapters import NOAAAdapter, OpenMeteoAdapter, WeatherFeatures
    
    # Initialize weather adapters
    noaa_config = {
        'api_key': None,  # NOAA doesn't require API key
        'base_url': 'https://api.weather.gov',
        'timeout': 30,
        'max_retries': 3
    }
    
    openmeteo_config = {
        'base_url': 'https://api.open-meteo.com/v1/forecast',
        'timeout': 30,
        'max_retries': 3
    }
    
    # Log data source access
    log_data_source_access(
        source_name="NOAA_NWS_API",
        url="https://api.weather.gov/gridpoints",
        data_type="weather_forecast",
        logger=logger,
        coverage_area="MISO footprint",
        temporal_range="48h forecast",
        cache_status="fresh"
    )
    
    log_data_source_access(
        source_name="OpenMeteo_API", 
        url="https://api.open-meteo.com/v1/forecast",
        data_type="weather_forecast_fallback",
        logger=logger,
        coverage_area="Global",
        temporal_range="48h forecast",
        cache_status="fresh"
    )
    
    # Process weather data for each horizon
    weather_data_list = []
    horizons = [12, 24, 36, 48]
    
    for horizon in horizons:
        with performance_monitor(f"weather_processing_{horizon}h", logger, input_size=len(hex_grid)):
            
            try:
                # Try NOAA first
                noaa_adapter = NOAAAdapter(noaa_config, cache_dir="data/raw")
                weather_data = noaa_adapter.fetch_forecast(horizon, hex_grid)
                
                logger.info(f"NOAA data fetched for {horizon}h horizon: {len(weather_data)} records",
                           component="weather_ingestion")
                
            except Exception as e:
                logger.warning(f"NOAA API failed for {horizon}h: {str(e)}", 
                              component="weather_ingestion")
                
                # Fallback to Open-Meteo
                try:
                    openmeteo_adapter = OpenMeteoAdapter(openmeteo_config, cache_dir="data/raw")
                    weather_data = openmeteo_adapter.fetch_forecast(horizon, hex_grid)
                    
                    logger.info(f"Open-Meteo fallback successful for {horizon}h: {len(weather_data)} records",
                               component="weather_ingestion")
                    
                except Exception as e2:
                    logger.error(f"Both weather APIs failed for {horizon}h: NOAA={str(e)}, OpenMeteo={str(e2)}",
                                component="weather_ingestion")
                    continue
            
            weather_data_list.append(weather_data)
    
    # Combine all weather data
    if weather_data_list:
        combined_weather = pd.concat(weather_data_list, ignore_index=True)
        
        # Save weather data
        weather_output_path = "data/processed/weather_features.csv"
        combined_weather.to_csv(weather_output_path, index=False)
        
        logger.info(f"Combined weather data saved: {len(combined_weather)} records to {weather_output_path}",
                   component="data_export")
    else:
        logger.error("No weather data successfully retrieved", component="weather_ingestion")

# =============================================================================
# CELL 4: Risk Scoring with Performance Monitoring
# =============================================================================

with log_step(
    step_name="Risk Score Calculation",
    logger=logger,
    doc_gen=doc_gen,
    description="Calculate comprehensive risk scores using weather, infrastructure, and exposure data",
    inputs=["weather_features.csv", "infrastructure_data", "population_data"],
    outputs=["risk_scores.csv", "confidence_metrics.csv"],
    methodology="Weighted combination: Risk = zscore(α×Hazard + β×Exposure + γ×Vulnerability)",
    requirements=["4.4", "5.4", "6.1"],
    assumptions=["All data sources have adequate coverage", "Weights properly calibrated"],
    limitations=["Z-score normalization may not capture extreme events", "Static weight configuration"]
):
    
    # Import risk scoring components
    with performance_monitor("risk_scoring_imports", logger):
        from risk_scoring_engine import RiskScoringEngine, RiskWeights
        from feature_engineering import FeatureEngineer
    
    # Initialize components
    weights = RiskWeights(
        thermal=0.3, wind=0.3, precip=0.25, storm=0.15,
        pop=0.7, load=0.3,
        renew_share=0.6, tx_scarcity=0.3, outage=0.1,
        alpha=0.5, beta=0.3, gamma=0.2
    )
    
    feature_engineer = FeatureEngineer()
    risk_engine = RiskScoringEngine(weights)
    
    # Process features
    with performance_monitor("feature_engineering", logger, input_size=len(combined_weather)):
        # Engineer weather features
        weather_features = feature_engineer.engineer_weather_features(combined_weather)
        
        logger.info(f"Weather features engineered: {len(weather_features)} records",
                   component="feature_engineering")
    
    # Calculate risk scores
    with performance_monitor("risk_calculation", logger, input_size=len(weather_features)):
        risk_scores = risk_engine.calculate_risk_scores(
            weather_features=weather_features,
            infrastructure_data=None,  # Would load actual infrastructure data
            exposure_data=None  # Would load actual exposure data
        )
        
        logger.info(f"Risk scores calculated: {len(risk_scores)} records",
                   component="risk_scoring")
    
    # Save risk scores
    risk_output_path = "data/processed/risk_scores.csv"
    risk_scores.to_csv(risk_output_path, index=False)
    
    logger.info(f"Risk scores saved to: {risk_output_path}", component="data_export")

# =============================================================================
# CELL 5: Generate Final Reports and Summary
# =============================================================================

# Generate comprehensive execution summary
final_summary = generate_notebook_summary(
    logger=logger,
    doc_gen=doc_gen,
    save_reports=True,
    display_summary=True
)

# Display performance analysis
perf_df = logger.get_performance_summary()
if not perf_df.empty:
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Top 5 slowest operations
    top_slow = perf_df.nlargest(5, 'duration_ms')
    print("\nTop 5 Slowest Operations:")
    for _, row in top_slow.iterrows():
        print(f"  {row['operation']}: {row['duration_ms']:.1f}ms "
              f"(Memory: {row['memory_delta_mb']:+.1f}MB)")
    
    # Memory usage analysis
    total_memory_delta = perf_df['memory_delta_mb'].sum()
    max_memory_usage = perf_df['memory_end_mb'].max()
    
    print(f"\nMemory Analysis:")
    print(f"  Total Memory Delta: {total_memory_delta:+.1f}MB")
    print(f"  Peak Memory Usage: {max_memory_usage:.1f}MB")
    
    # Operation categories
    spatial_ops = perf_df[perf_df['operation'].str.contains('spatial|grid|hex', case=False)]
    weather_ops = perf_df[perf_df['operation'].str.contains('weather|api|fetch', case=False)]
    risk_ops = perf_df[perf_df['operation'].str.contains('risk|scoring|feature', case=False)]
    
    print(f"\nOperation Categories:")
    if not spatial_ops.empty:
        print(f"  Spatial Operations: {len(spatial_ops)} ops, {spatial_ops['duration_ms'].sum():.1f}ms total")
    if not weather_ops.empty:
        print(f"  Weather Operations: {len(weather_ops)} ops, {weather_ops['duration_ms'].sum():.1f}ms total")
    if not risk_ops.empty:
        print(f"  Risk Operations: {len(risk_ops)} ops, {risk_ops['duration_ms'].sum():.1f}ms total")

# Display data source summary
print(f"\nData Sources Accessed: {len(logger.data_sources)}")
for source in logger.data_sources:
    status_icon = "✅" if not source.error_status else "❌"
    print(f"  {status_icon} {source.source_name} ({source.data_type})")
    if source.error_status:
        print(f"    Error: {source.error_status}")

# Display final file exports
if 'exported_files' in final_summary['execution_summary']:
    exported = final_summary['execution_summary']['exported_files']
    print(f"\nExported Files:")
    print(f"  Documentation: {exported['documentation']}")
    for format_type, file_path in exported['logs'].items():
        print(f"  {format_type.upper()} logs: {file_path}")

print(f"\n✓ Notebook execution completed successfully!")
print(f"Session ID: {logger.session_id}")
print(f"Total Duration: {final_summary['execution_summary']['total_duration_minutes']:.1f} minutes")

# Close logger
logger.close()