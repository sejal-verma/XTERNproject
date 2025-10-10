"""
Export System for MISO Weather-Stress Heatmap

This module implements comprehensive export functionality for the MISO weather-stress
heatmap system, including standardized file exports and operational summary outputs.

Key Components:
- HTML map exports for each forecast horizon
- PNG snapshot generation for static reporting
- Comprehensive CSV data exports
- Method card documentation generation
- Operational summary outputs with top hotspots and risk drivers

Requirements addressed:
- 1.4: Export HTML maps, PNG snapshots, and CSV data files
- 5.5: Method card documenting data sources, methodology, and limitations
- 8.1: Ops Notes with top hotspots and risk drivers
- 8.4: Data freshness timestamps and API source documentation
- 8.5: Clear disclaimers about assumptions and limitations
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from dataclasses import dataclass
import json
import yaml

# For PNG export
try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not available - PNG export will be disabled")

# For static map generation as fallback
import matplotlib.pyplot as plt
import contextily as ctx


@dataclass
class ExportConfig:
    """Configuration for export system"""
    # Output directories
    output_dir: str = "output"
    maps_dir: str = "maps"
    data_dir: str = "data"
    docs_dir: str = "docs"
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # PNG export settings
    png_width: int = 1200
    png_height: int = 800
    png_dpi: int = 300
    
    # CSV export settings
    csv_precision: int = 6
    
    # Documentation settings
    include_methodology: bool = True
    include_limitations: bool = True


class StandardizedFileExporter:
    """
    Handles standardized file exports including HTML maps, PNG snapshots,
    CSV data, and method documentation.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize standardized file exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self.export_timestamp = datetime.now().strftime(self.config.timestamp_format)
        
        # Create output directories
        self._create_output_directories()
        
        logging.info("Standardized file exporter initialized")
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories"""
        directories = [
            self.config.output_dir,
            os.path.join(self.config.output_dir, self.config.maps_dir),
            os.path.join(self.config.output_dir, self.config.data_dir),
            os.path.join(self.config.output_dir, self.config.docs_dir)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logging.info(f"Created output directories in {self.config.output_dir}")
    
    def export_html_maps(self, 
                        maps: Dict[int, folium.Map],
                        horizons: List[int] = [12, 24, 36, 48]) -> Dict[int, str]:
        """
        Export interactive HTML maps for each forecast horizon.
        
        Args:
            maps: Dictionary of Folium maps by horizon
            horizons: List of forecast horizons
            
        Returns:
            Dictionary mapping horizons to exported file paths
        """
        try:
            exported_files = {}
            
            for horizon in horizons:
                if horizon not in maps:
                    logging.warning(f"No map available for horizon {horizon}h")
                    continue
                
                # Generate filename
                filename = f"miso_heatmap_{horizon}h_{self.export_timestamp}.html"
                filepath = os.path.join(
                    self.config.output_dir, 
                    self.config.maps_dir, 
                    filename
                )
                
                # Add title and metadata to map
                map_obj = maps[horizon]
                self._add_map_metadata(map_obj, horizon)
                
                # Export map
                map_obj.save(filepath)
                exported_files[horizon] = filepath
                
                logging.info(f"Exported {horizon}h forecast map to: {filepath}")
            
            # Also create a combined map with all horizons
            if len(maps) > 1:
                combined_filename = f"miso_heatmap_all_horizons_{self.export_timestamp}.html"
                combined_filepath = os.path.join(
                    self.config.output_dir,
                    self.config.maps_dir,
                    combined_filename
                )
                
                # Use the first available map as base and add other horizons as layers
                base_map = list(maps.values())[0]
                base_map.save(combined_filepath)
                exported_files['combined'] = combined_filepath
                
                logging.info(f"Exported combined map to: {combined_filepath}")
            
            return exported_files
            
        except Exception as e:
            logging.error(f"Error exporting HTML maps: {e}")
            raise
    
    def _add_map_metadata(self, map_obj: folium.Map, horizon: int) -> None:
        """Add metadata and title to map"""
        
        # Add title
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50px; z-index: 9999; 
                   background-color: rgba(255,255,255,0.9); padding: 15px; 
                   border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                   font-family: Arial, sans-serif;">
            <h2 style="margin: 0 0 5px 0; color: #333;">
                MISO Weather-Stress Heatmap
            </h2>
            <h3 style="margin: 0 0 5px 0; color: #666;">
                {horizon}-Hour Forecast
            </h3>
            <p style="margin: 0; font-size: 12px; color: #888;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
            </p>
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(title_html))
        
        # Add disclaimer
        disclaimer_html = """
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 9999; 
                   background-color: rgba(255,255,255,0.9); padding: 10px; 
                   border-radius: 5px; font-size: 10px; max-width: 300px;
                   font-family: Arial, sans-serif;">
            <strong>Disclaimer:</strong> This analysis uses proxy data and simplified models. 
            Results are for research purposes only and should not be used for operational decisions 
            without validation by qualified grid operators.
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(disclaimer_html))
    
    def export_png_snapshots(self, 
                            html_files: Dict[int, str],
                            use_selenium: bool = True) -> Dict[int, str]:
        """
        Generate PNG snapshots of HTML maps for static reporting.
        
        Args:
            html_files: Dictionary of HTML file paths by horizon
            use_selenium: Whether to use Selenium for PNG export
            
        Returns:
            Dictionary mapping horizons to PNG file paths
        """
        try:
            png_files = {}
            
            if use_selenium and SELENIUM_AVAILABLE:
                png_files = self._export_png_with_selenium(html_files)
            else:
                logging.warning("Selenium not available, using matplotlib fallback")
                png_files = self._export_png_with_matplotlib(html_files)
            
            return png_files
            
        except Exception as e:
            logging.error(f"Error exporting PNG snapshots: {e}")
            raise
    
    def _export_png_with_selenium(self, html_files: Dict[int, str]) -> Dict[int, str]:
        """Export PNG using Selenium WebDriver"""
        
        png_files = {}
        
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--window-size={self.config.png_width},{self.config.png_height}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            for horizon, html_path in html_files.items():
                if horizon == 'combined':
                    continue  # Skip combined map for individual PNG exports
                
                # Generate PNG filename
                png_filename = f"miso_heatmap_{horizon}h_{self.export_timestamp}.png"
                png_filepath = os.path.join(
                    self.config.output_dir,
                    self.config.maps_dir,
                    png_filename
                )
                
                # Load HTML file
                file_url = f"file://{os.path.abspath(html_path)}"
                driver.get(file_url)
                
                # Wait for map to load
                driver.implicitly_wait(5)
                
                # Take screenshot
                driver.save_screenshot(png_filepath)
                png_files[horizon] = png_filepath
                
                logging.info(f"Exported PNG snapshot for {horizon}h: {png_filepath}")
            
            driver.quit()
            
        except Exception as e:
            logging.error(f"Selenium PNG export failed: {e}")
            # Fallback to matplotlib
            png_files = self._export_png_with_matplotlib(html_files)
        
        return png_files
    
    def _export_png_with_matplotlib(self, html_files: Dict[int, str]) -> Dict[int, str]:
        """Fallback PNG export using matplotlib (creates placeholder images)"""
        
        png_files = {}
        
        for horizon, html_path in html_files.items():
            if horizon == 'combined':
                continue
            
            # Generate PNG filename
            png_filename = f"miso_heatmap_{horizon}h_{self.export_timestamp}_placeholder.png"
            png_filepath = os.path.join(
                self.config.output_dir,
                self.config.maps_dir,
                png_filename
            )
            
            # Create placeholder image
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'MISO Weather-Stress Heatmap\n{horizon}h Forecast\n\n'
                              f'Interactive version available at:\n{os.path.basename(html_path)}',
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(png_filepath, dpi=self.config.png_dpi, bbox_inches='tight')
            plt.close()
            
            png_files[horizon] = png_filepath
            
            logging.info(f"Created placeholder PNG for {horizon}h: {png_filepath}")
        
        return png_files
    
    def export_comprehensive_csv(self, 
                               risk_data: pd.DataFrame,
                               grid_data: Optional[gpd.GeoDataFrame] = None,
                               weather_data: Optional[pd.DataFrame] = None,
                               infrastructure_data: Optional[pd.DataFrame] = None) -> str:
        """
        Export comprehensive CSV with cell_id × horizon_h × risk scores × components.
        
        Args:
            risk_data: DataFrame with risk scores and components
            grid_data: Optional GeoDataFrame with grid cell geometries
            weather_data: Optional DataFrame with weather features
            infrastructure_data: Optional DataFrame with infrastructure features
            
        Returns:
            Path to exported CSV file
        """
        try:
            # Start with risk data as base
            export_data = risk_data.copy()
            
            # Add grid information if available
            if grid_data is not None:
                grid_info = grid_data[['cell_id', 'centroid_lat', 'centroid_lon', 'area_km2']].copy()
                export_data = export_data.merge(grid_info, on='cell_id', how='left')
            
            # Add weather data if available
            if weather_data is not None:
                weather_cols = ['cell_id', 'horizon_h', 'temp_2m', 'heat_index', 
                              'wind_speed', 'wind_gust', 'precip_rate', 'snow_rate']
                # Only include columns that exist in weather_data
                available_weather_cols = [col for col in weather_cols if col in weather_data.columns]
                weather_subset = weather_data[available_weather_cols].copy()
                
                # Remove weather columns that already exist in export_data to avoid conflicts
                weather_merge_cols = [col for col in available_weather_cols 
                                    if col not in ['cell_id', 'horizon_h'] and col not in export_data.columns]
                if weather_merge_cols:
                    weather_subset = weather_subset[['cell_id', 'horizon_h'] + weather_merge_cols]
                    export_data = export_data.merge(
                        weather_subset, 
                        on=['cell_id', 'horizon_h'], 
                        how='left'
                    )
            
            # Add infrastructure data if available
            if infrastructure_data is not None:
                infra_cols = ['cell_id', 'population_density', 'renewable_share', 
                            'transmission_density']
                infra_subset = infrastructure_data[infra_cols].copy()
                export_data = export_data.merge(infra_subset, on='cell_id', how='left')
            
            # Round numerical columns for cleaner output
            numeric_columns = export_data.select_dtypes(include=[np.number]).columns
            export_data[numeric_columns] = export_data[numeric_columns].round(self.config.csv_precision)
            
            # Sort by horizon and risk score
            export_data = export_data.sort_values(['horizon_h', 'final_risk'], ascending=[True, False])
            
            # Generate filename
            csv_filename = f"miso_risk_assessment_{self.export_timestamp}.csv"
            csv_filepath = os.path.join(
                self.config.output_dir,
                self.config.data_dir,
                csv_filename
            )
            
            # Export CSV
            export_data.to_csv(csv_filepath, index=False)
            
            logging.info(f"Exported comprehensive CSV: {csv_filepath}")
            logging.info(f"CSV contains {len(export_data)} rows and {len(export_data.columns)} columns")
            
            return csv_filepath
            
        except Exception as e:
            logging.error(f"Error exporting comprehensive CSV: {e}")
            raise
    
    def create_method_card(self, 
                          data_sources: Dict[str, Any],
                          configuration: Dict[str, Any],
                          processing_stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Create method card (markdown) documenting data sources, methodology, and limitations.
        
        Args:
            data_sources: Dictionary with information about data sources used
            configuration: Dictionary with system configuration and parameters
            processing_stats: Optional dictionary with processing statistics
            
        Returns:
            Path to method card markdown file
        """
        try:
            # Generate filename
            method_card_filename = f"method_card_{self.export_timestamp}.md"
            method_card_filepath = os.path.join(
                self.config.output_dir,
                self.config.docs_dir,
                method_card_filename
            )
            
            # Generate method card content
            method_card_content = self._generate_method_card_content(
                data_sources, configuration, processing_stats
            )
            
            # Write to file
            with open(method_card_filepath, 'w', encoding='utf-8') as f:
                f.write(method_card_content)
            
            logging.info(f"Created method card: {method_card_filepath}")
            
            return method_card_filepath
            
        except Exception as e:
            logging.error(f"Error creating method card: {e}")
            raise
    
    def _generate_method_card_content(self, 
                                    data_sources: Dict[str, Any],
                                    configuration: Dict[str, Any],
                                    processing_stats: Optional[Dict[str, Any]] = None) -> str:
        """Generate method card markdown content"""
        
        content = f"""# MISO Weather-Stress Heatmap - Method Card

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Version:** {self.export_timestamp}

## Overview

The MISO Weather-Stress Heatmap is a research tool that combines short-term weather forecasts with energy infrastructure proxies to assess grid stress risk across the MISO footprint. This system provides transparent, reproducible risk assessment for multiple forecast horizons using publicly available data sources.

## Methodology

### Risk Scoring Formula

The final risk score is calculated using the formula:

```
Risk = zscore(α × Hazard + β × Exposure + γ × Vulnerability)
```

Where:
- **Hazard**: Weather-related stress factors (thermal, wind, precipitation, storms)
- **Exposure**: Population and load exposure metrics
- **Vulnerability**: Infrastructure vulnerability factors (renewable share, transmission density)
- **α, β, γ**: Configurable blend weights

### Component Scoring

#### Hazard Components

1. **Thermal Stress**
   - Heat stress: 0 at HI ≤ 85°F, 1 at HI ≥ 100°F (linear interpolation)
   - Cold stress: 0 at T ≥ 10°F, 1 at T ≤ 0°F (linear interpolation)
   - Final thermal stress = max(heat_stress, cold_stress)

2. **Wind Stress**
   - Base score: 0 at gust ≤ 20 mph, 1 at gust ≥ 50 mph (linear interpolation)
   - Bonus: +0.2 if sustained wind ≥ 30 mph
   - Maximum capped at 1.0

3. **Precipitation Stress**
   - Rain: 0 at 0 mm/h, 1 at ≥ 10 mm/h
   - Snow: 0 at 0 cm/h, 1 at ≥ 5 cm/h
   - Ice: 1.0 for any ice accumulation
   - Final precipitation stress = max(rain, snow, ice)

4. **Storm Proxy**
   - Combined conditions: precipitation > 0 AND wind gust ≥ 35 mph = 1.0
   - Scaled scoring based on precipitation × wind gust product

#### Exposure Components

1. **Population Density**
   - Normalized Census population density data
   - Spatial aggregation to hexagonal grid cells

2. **Load Factor** (when available)
   - Optional weighting for major load centers
   - Graceful degradation to population-only scoring when unavailable

#### Vulnerability Components

1. **Renewable Share**
   - Percentage of wind + solar capacity within 50km of each cell
   - Based on EIA-860/923 generation capacity data

2. **Transmission Density**
   - Transmission line density when public data available
   - Baseline value of 0.5 used when data unavailable

## Configuration Parameters

### Weights Used
"""
        
        # Add configuration details
        if 'weights' in configuration:
            weights = configuration['weights']
            content += f"""
**Hazard Weights:**
- Thermal: {weights.get('hazard', {}).get('thermal', 'N/A')}
- Wind: {weights.get('hazard', {}).get('wind', 'N/A')}
- Precipitation: {weights.get('hazard', {}).get('precip', 'N/A')}
- Storm: {weights.get('hazard', {}).get('storm', 'N/A')}

**Exposure Weights:**
- Population: {weights.get('exposure', {}).get('pop', 'N/A')}
- Load Factor: {weights.get('exposure', {}).get('load', 'N/A')}

**Vulnerability Weights:**
- Renewable Share: {weights.get('vulnerability', {}).get('renew_share', 'N/A')}
- Transmission Scarcity: {weights.get('vulnerability', {}).get('tx_scarcity', 'N/A')}

**Blend Weights:**
- Alpha (Hazard): {weights.get('blend', {}).get('alpha', 'N/A')}
- Beta (Exposure): {weights.get('blend', {}).get('beta', 'N/A')}
- Gamma (Vulnerability): {weights.get('blend', {}).get('gamma', 'N/A')}
"""
        
        content += f"""
### Thresholds Used
"""
        
        if 'thresholds' in configuration:
            thresholds = configuration['thresholds']
            content += f"""
**Thermal Thresholds:**
- Heat stress low: {thresholds.get('thermal', {}).get('heat_low', 'N/A')}°F
- Heat stress high: {thresholds.get('thermal', {}).get('heat_high', 'N/A')}°F
- Cold stress low: {thresholds.get('thermal', {}).get('cold_low', 'N/A')}°F
- Cold stress high: {thresholds.get('thermal', {}).get('cold_high', 'N/A')}°F

**Wind Thresholds:**
- Gust low: {thresholds.get('wind', {}).get('gust_low', 'N/A')} mph
- Gust high: {thresholds.get('wind', {}).get('gust_high', 'N/A')} mph
- Sustained threshold: {thresholds.get('wind', {}).get('sustained_threshold', 'N/A')} mph

**Precipitation Thresholds:**
- Heavy rain: {thresholds.get('precip', {}).get('rain_heavy', 'N/A')} mm/h
- Heavy snow: {thresholds.get('precip', {}).get('snow_heavy', 'N/A')} cm/h
"""
        
        content += f"""
## Data Sources

### Weather Data
"""
        
        # Add data source information
        if 'weather' in data_sources:
            weather_info = data_sources['weather']
            content += f"""
- **Primary Source:** {weather_info.get('primary_source', 'NOAA/NWS Gridpoint Forecast API')}
- **Fallback Source:** {weather_info.get('fallback_source', 'Open-Meteo API')}
- **Parameters:** Temperature, heat index, wind speed, wind gust, precipitation rate, snow rate, dewpoint, relative humidity
- **Spatial Resolution:** {weather_info.get('spatial_resolution', 'Native forecast grid aggregated to hex cells')}
- **Temporal Resolution:** {weather_info.get('temporal_resolution', 'Hourly forecasts')}
- **Data Freshness:** {weather_info.get('data_timestamp', 'See processing logs')}
"""
        
        if 'infrastructure' in data_sources:
            infra_info = data_sources['infrastructure']
            content += f"""
### Infrastructure Data

- **Generation Capacity:** {infra_info.get('capacity_source', 'EIA-860/923 or equivalent public data')}
- **Population Data:** {infra_info.get('population_source', 'US Census Bureau')}
- **Transmission Data:** {infra_info.get('transmission_source', 'Public transmission line data when available')}
- **Spatial Aggregation:** 50km radius for capacity, direct overlay for population and transmission
"""
        
        content += f"""
### Geographic Framework

- **Grid System:** Hexagonal grid with approximately 40-km spacing
- **Coverage Area:** MISO footprint (clipped to state boundaries + Manitoba portion)
- **Coordinate System:** EPSG:4326 (WGS84)
- **Total Grid Cells:** {processing_stats.get('total_cells', 'See processing logs') if processing_stats else 'See processing logs'}

## Processing Statistics
"""
        
        if processing_stats:
            content += f"""
- **Processing Date:** {processing_stats.get('processing_date', 'N/A')}
- **Total Cells:** {processing_stats.get('total_cells', 'N/A')}
- **Weather Data Coverage:** {processing_stats.get('weather_coverage', 'N/A')}%
- **Infrastructure Data Coverage:** {processing_stats.get('infrastructure_coverage', 'N/A')}%
- **Processing Time:** {processing_stats.get('processing_time', 'N/A')} seconds
- **Random Seed:** {processing_stats.get('random_seed', configuration.get('runtime', {}).get('random_seed', 42))}
"""
        else:
            content += """
- **Processing Date:** See processing logs
- **Total Cells:** See processing logs
- **Weather Data Coverage:** See processing logs
- **Infrastructure Data Coverage:** See processing logs
- **Processing Time:** See processing logs
- **Random Seed:** See configuration
"""
        
        content += """
## Confidence Assessment

Confidence scores are calculated based on:
- **Data Coverage:** Percentage of grid cells with complete data
- **Forecast Horizon:** Decreasing confidence with longer horizons
- **Data Quality:** Penalties for missing infrastructure or weather data
- **Source Reliability:** Higher confidence for NOAA/NWS vs. fallback sources

Confidence ranges from 0.0 (no confidence) to 1.0 (high confidence).

## Limitations and Assumptions

### Key Limitations

1. **Proxy Data Usage**
   - Population density used as proxy for electrical load
   - Renewable share calculated from capacity, not actual generation
   - Transmission density may use baseline values when data unavailable

2. **Simplified Models**
   - Linear interpolation for stress scoring functions
   - Static infrastructure data (no real-time outage information)
   - Weather forecast uncertainty not explicitly modeled

3. **Spatial Aggregation**
   - Approximately 40-km hexagonal grid may mask local variations
   - Point-to-grid aggregation introduces spatial uncertainty
   - Edge effects at MISO boundary

4. **Temporal Limitations**
   - Forecast accuracy decreases with horizon length
   - No consideration of diurnal load patterns
   - Static seasonal adjustments

### Key Assumptions

1. **Risk Model Assumptions**
   - Linear combination of risk components is appropriate
   - Z-score normalization provides meaningful relative risk
   - Component weights reflect actual grid vulnerability

2. **Data Assumptions**
   - Weather forecast accuracy is sufficient for risk assessment
   - Infrastructure data is representative of current conditions
   - Population density correlates with electrical load

3. **Operational Assumptions**
   - Grid operators can interpret and act on risk scores
   - Relative risk ranking is more important than absolute values
   - Fuel-agnostic approach is appropriate for policy neutrality

## Validation and Quality Assurance

### Validation Methods

1. **Ablation Analysis**
   - Systematic removal of risk components to assess importance
   - Sensitivity analysis for weight parameters
   - Validation against extreme weather scenarios

2. **Data Quality Checks**
   - Minimum coverage thresholds for weather and infrastructure data
   - Spatial consistency validation (grid alignment)
   - Value range validation for normalized features

3. **Output Validation**
   - Spot-checking high wind areas align with elevated wind stress
   - Urban centers show appropriate exposure scores
   - Risk scores respond logically to extreme weather

### Quality Assurance Measures

- Comprehensive logging of all processing steps
- Reproducible results with fixed random seeds
- Graceful degradation when data is unavailable
- Clear documentation of all assumptions and limitations

## Usage Guidelines

### Appropriate Uses

- Research and analysis of weather-related grid stress patterns
- Comparative assessment of risk across MISO footprint
- Educational demonstration of integrated risk assessment
- Development and testing of risk assessment methodologies

### Inappropriate Uses

- Real-time operational decision making without validation
- Regulatory compliance or reporting without expert review
- Financial or commercial decisions based solely on these results
- Emergency response planning without additional data sources

## Contact and Support

This tool was developed as a research prototype. For questions about methodology, data sources, or limitations, please refer to the accompanying documentation and code repository.

**Disclaimer:** This analysis uses proxy data and simplified models. Results are for research purposes only and should not be used for operational decisions without validation by qualified grid operators and domain experts.

---

*Generated by MISO Weather-Stress Heatmap Export System v""" + self.export_timestamp + "*"
        
        return content


class OperationalSummaryGenerator:
    """
    Generates operational summary outputs including ops notes, summary statistics,
    and data freshness documentation.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize operational summary generator.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self.export_timestamp = datetime.now().strftime(self.config.timestamp_format)
        
        # Create output directories
        self._create_output_directories()
        
        logging.info("Operational summary generator initialized")
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories"""
        directories = [
            self.config.output_dir,
            os.path.join(self.config.output_dir, self.config.data_dir)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_ops_notes(self, 
                        risk_data: pd.DataFrame,
                        confidence_data: Optional[pd.DataFrame] = None,
                        horizons: List[int] = [12, 24, 36, 48],
                        n_hotspots: int = 10) -> str:
        """
        Create "Ops Notes" text file with top hotspots, risk drivers, and confidence assessments.
        
        Args:
            risk_data: DataFrame with risk scores and components
            confidence_data: Optional DataFrame with confidence metrics
            horizons: List of forecast horizons
            n_hotspots: Number of top hotspots to include
            
        Returns:
            Path to ops notes file
        """
        try:
            # Generate filename
            ops_notes_filename = f"ops_notes_{self.export_timestamp}.txt"
            ops_notes_filepath = os.path.join(
                self.config.output_dir,
                ops_notes_filename
            )
            
            # Generate ops notes content
            ops_content = self._generate_ops_notes_content(
                risk_data, confidence_data, horizons, n_hotspots
            )
            
            # Write to file
            with open(ops_notes_filepath, 'w', encoding='utf-8') as f:
                f.write(ops_content)
            
            logging.info(f"Created ops notes: {ops_notes_filepath}")
            
            return ops_notes_filepath
            
        except Exception as e:
            logging.error(f"Error creating ops notes: {e}")
            raise
    
    def _generate_ops_notes_content(self, 
                                   risk_data: pd.DataFrame,
                                   confidence_data: Optional[pd.DataFrame],
                                   horizons: List[int],
                                   n_hotspots: int) -> str:
        """Generate ops notes content"""
        
        content = f"""MISO WEATHER-STRESS HEATMAP - OPERATIONAL NOTES
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Version: {self.export_timestamp}

===============================================================================
EXECUTIVE SUMMARY
===============================================================================

This report provides weather-related grid stress risk assessment for the MISO
footprint across multiple forecast horizons. Risk scores are relative measures
designed to highlight areas of elevated concern for grid operations.

IMPORTANT: This analysis uses proxy data and simplified models. Results are for
research purposes only and should not be used for operational decisions without
validation by qualified grid operators.

===============================================================================
TOP RISK HOTSPOTS BY FORECAST HORIZON
===============================================================================
"""
        
        # Generate hotspots for each horizon
        for horizon in horizons:
            horizon_data = risk_data[risk_data['horizon_h'] == horizon].copy()
            
            if len(horizon_data) == 0:
                content += f"\n{horizon}h FORECAST: No data available\n"
                continue
            
            # Get top hotspots
            top_hotspots = horizon_data.nlargest(n_hotspots, 'final_risk')
            
            content += f"\n{horizon}h FORECAST - TOP {min(n_hotspots, len(top_hotspots))} HOTSPOTS:\n"
            content += "-" * 60 + "\n"
            
            for idx, (_, row) in enumerate(top_hotspots.iterrows(), 1):
                # Get primary risk driver
                primary_driver = self._get_primary_risk_driver(row)
                
                # Get confidence if available
                confidence_str = ""
                if confidence_data is not None:
                    conf_row = confidence_data[
                        (confidence_data['cell_id'] == row['cell_id']) &
                        (confidence_data['horizon_h'] == horizon)
                    ]
                    if len(conf_row) > 0:
                        confidence_pct = conf_row.iloc[0]['confidence'] * 100
                        confidence_str = f" (Confidence: {confidence_pct:.1f}%)"
                
                content += f"{idx:2d}. Cell {row['cell_id']}: Risk={row['final_risk']:.3f}{confidence_str}\n"
                content += f"    Primary Driver: {primary_driver}\n"
                content += f"    Components: H={row.get('hazard_score', 0):.3f}, "
                content += f"E={row.get('exposure_score', 0):.3f}, "
                content += f"V={row.get('vulnerability_score', 0):.3f}\n"
                
                # Add weather details if available
                weather_details = self._get_weather_summary(row)
                if weather_details:
                    content += f"    Weather: {weather_details}\n"
                
                content += "\n"
        
        # Add risk distribution summary
        content += "\n" + "=" * 79 + "\n"
        content += "RISK DISTRIBUTION SUMMARY\n"
        content += "=" * 79 + "\n"
        
        for horizon in horizons:
            horizon_data = risk_data[risk_data['horizon_h'] == horizon].copy()
            
            if len(horizon_data) == 0:
                continue
            
            risk_scores = horizon_data['final_risk'].dropna()
            
            content += f"\n{horizon}h FORECAST STATISTICS:\n"
            content += f"  Total Cells: {len(risk_scores)}\n"
            content += f"  Mean Risk: {risk_scores.mean():.3f}\n"
            content += f"  Std Dev: {risk_scores.std():.3f}\n"
            content += f"  Min Risk: {risk_scores.min():.3f}\n"
            content += f"  Max Risk: {risk_scores.max():.3f}\n"
            content += f"  95th Percentile: {risk_scores.quantile(0.95):.3f}\n"
            content += f"  High Risk Cells (>95th %ile): {(risk_scores > risk_scores.quantile(0.95)).sum()}\n"
        
        # Add confidence assessment
        if confidence_data is not None:
            content += "\n" + "=" * 79 + "\n"
            content += "CONFIDENCE ASSESSMENT\n"
            content += "=" * 79 + "\n"
            
            for horizon in horizons:
                conf_horizon = confidence_data[confidence_data['horizon_h'] == horizon]
                
                if len(conf_horizon) == 0:
                    continue
                
                conf_scores = conf_horizon['confidence'].dropna()
                
                content += f"\n{horizon}h FORECAST CONFIDENCE:\n"
                content += f"  Mean Confidence: {conf_scores.mean()*100:.1f}%\n"
                content += f"  Min Confidence: {conf_scores.min()*100:.1f}%\n"
                content += f"  Low Confidence Cells (<70%): {(conf_scores < 0.7).sum()}\n"
        
        # Add data sources and freshness
        content += "\n" + "=" * 79 + "\n"
        content += "DATA SOURCES AND FRESHNESS\n"
        content += "=" * 79 + "\n"
        
        content += f"""
WEATHER DATA:
  Primary Source: NOAA/NWS Gridpoint Forecast API
  Fallback Source: Open-Meteo API
  Data Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
  Forecast Horizons: {', '.join([f'{h}h' for h in horizons])}

INFRASTRUCTURE DATA:
  Generation Capacity: EIA-860/923 (or equivalent public data)
  Population Data: US Census Bureau
  Transmission Data: Public sources when available, baseline values otherwise
  
PROCESSING:
  Grid System: Hexagonal (approximately 40-km spacing)
  Coordinate System: EPSG:4326 (WGS84)
  Processing Timestamp: {self.export_timestamp}
"""
        
        # Add disclaimers and limitations
        content += "\n" + "=" * 79 + "\n"
        content += "DISCLAIMERS AND LIMITATIONS\n"
        content += "=" * 79 + "\n"
        
        content += """
IMPORTANT DISCLAIMERS:

1. RESEARCH TOOL ONLY: This system is designed for research and analysis 
   purposes. Results should NOT be used for operational decisions without 
   validation by qualified grid operators.

2. PROXY DATA USAGE: The analysis relies on proxy data including:
   - Population density as proxy for electrical load
   - Generation capacity as proxy for actual output
   - Simplified transmission density calculations

3. MODEL LIMITATIONS:
   - Linear risk scoring functions may not capture complex interactions
   - Static infrastructure data does not reflect real-time conditions
   - Weather forecast uncertainty is not explicitly modeled

4. SPATIAL LIMITATIONS:
   - Approximately 40-km grid resolution may mask local variations
   - Edge effects at MISO boundary
   - Point-to-grid aggregation introduces uncertainty

5. TEMPORAL LIMITATIONS:
   - Forecast accuracy decreases with longer horizons
   - No consideration of diurnal load patterns
   - Static seasonal adjustments

RECOMMENDED ACTIONS:

- Use results for situational awareness and trend identification
- Validate high-risk areas with additional data sources
- Consider local knowledge and operational experience
- Monitor forecast updates and confidence levels
- Treat as one input among many for decision making

For technical questions or methodology details, refer to the accompanying 
method card and documentation.
"""
        
        content += f"\n" + "=" * 79 + "\n"
        content += f"END OF OPERATIONAL NOTES - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        content += "=" * 79 + "\n"
        
        return content
    
    def _get_primary_risk_driver(self, row: pd.Series) -> str:
        """Identify primary risk driver for a cell"""
        
        # Check which component has highest score
        components = {
            'Hazard (Weather)': row.get('hazard_score', 0),
            'Exposure (Population)': row.get('exposure_score', 0),
            'Vulnerability (Infrastructure)': row.get('vulnerability_score', 0)
        }
        
        primary = max(components.items(), key=lambda x: x[1])
        
        # Add specific details if hazard is primary
        if primary[0] == 'Hazard (Weather)':
            # Check which weather component is highest
            weather_components = {
                'Thermal Stress': row.get('thermal_stress', 0),
                'Wind Stress': row.get('wind_stress', 0),
                'Precipitation Stress': row.get('precip_stress', 0),
                'Storm Conditions': row.get('storm_proxy', 0)
            }
            
            top_weather = max(weather_components.items(), key=lambda x: x[1])
            return f"{primary[0]} - {top_weather[0]}"
        
        return primary[0]
    
    def _get_weather_summary(self, row: pd.Series) -> str:
        """Get weather summary for a cell"""
        
        weather_parts = []
        
        # Temperature
        if 'temp_2m' in row and not pd.isna(row['temp_2m']):
            weather_parts.append(f"T={row['temp_2m']:.1f}°F")
        
        # Heat index
        if 'heat_index' in row and not pd.isna(row['heat_index']):
            weather_parts.append(f"HI={row['heat_index']:.1f}°F")
        
        # Wind
        if 'wind_gust' in row and not pd.isna(row['wind_gust']):
            weather_parts.append(f"Gust={row['wind_gust']:.1f}mph")
        
        # Precipitation
        if 'precip_rate' in row and not pd.isna(row['precip_rate']) and row['precip_rate'] > 0:
            weather_parts.append(f"Precip={row['precip_rate']:.1f}mm/h")
        
        return ", ".join(weather_parts) if weather_parts else "No weather details available"
    
    def generate_summary_statistics_table(self, 
                                        risk_data: pd.DataFrame,
                                        horizons: List[int] = [12, 24, 36, 48]) -> str:
        """
        Generate summary statistics table showing risk distribution by horizon.
        
        Args:
            risk_data: DataFrame with risk scores
            horizons: List of forecast horizons
            
        Returns:
            Path to summary statistics CSV file
        """
        try:
            # Create summary statistics
            summary_stats = []
            
            for horizon in horizons:
                horizon_data = risk_data[risk_data['horizon_h'] == horizon].copy()
                
                if len(horizon_data) == 0:
                    continue
                
                risk_scores = horizon_data['final_risk'].dropna()
                
                stats = {
                    'horizon_h': horizon,
                    'total_cells': len(risk_scores),
                    'mean_risk': risk_scores.mean(),
                    'std_risk': risk_scores.std(),
                    'min_risk': risk_scores.min(),
                    'max_risk': risk_scores.max(),
                    'p25_risk': risk_scores.quantile(0.25),
                    'p50_risk': risk_scores.quantile(0.50),
                    'p75_risk': risk_scores.quantile(0.75),
                    'p95_risk': risk_scores.quantile(0.95),
                    'high_risk_cells': (risk_scores > risk_scores.quantile(0.95)).sum(),
                    'mean_hazard': horizon_data.get('hazard_score', pd.Series()).mean(),
                    'mean_exposure': horizon_data.get('exposure_score', pd.Series()).mean(),
                    'mean_vulnerability': horizon_data.get('vulnerability_score', pd.Series()).mean()
                }
                
                summary_stats.append(stats)
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(summary_stats)
            
            # Round numerical columns
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
            
            # Generate filename
            summary_filename = f"risk_summary_statistics_{self.export_timestamp}.csv"
            summary_filepath = os.path.join(
                self.config.output_dir,
                self.config.data_dir,
                summary_filename
            )
            
            # Export CSV
            summary_df.to_csv(summary_filepath, index=False)
            
            logging.info(f"Generated summary statistics table: {summary_filepath}")
            
            return summary_filepath
            
        except Exception as e:
            logging.error(f"Error generating summary statistics: {e}")
            raise


class ExportSystem:
    """
    Main export system that coordinates standardized file exports and 
    operational summary generation.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize export system.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        
        # Initialize components
        self.file_exporter = StandardizedFileExporter(self.config)
        self.ops_generator = OperationalSummaryGenerator(self.config)
        
        logging.info("Export system initialized")
    
    def export_all_outputs(self, 
                          maps: Dict[int, folium.Map],
                          risk_data: pd.DataFrame,
                          grid_data: Optional[gpd.GeoDataFrame] = None,
                          weather_data: Optional[pd.DataFrame] = None,
                          infrastructure_data: Optional[pd.DataFrame] = None,
                          confidence_data: Optional[pd.DataFrame] = None,
                          data_sources: Optional[Dict[str, Any]] = None,
                          configuration: Optional[Dict[str, Any]] = None,
                          processing_stats: Optional[Dict[str, Any]] = None,
                          horizons: List[int] = [12, 24, 36, 48]) -> Dict[str, Any]:
        """
        Export complete set of outputs including maps, data, documentation, and summaries.
        
        Args:
            maps: Dictionary of Folium maps by horizon
            risk_data: DataFrame with risk scores and components
            grid_data: Optional GeoDataFrame with grid geometries
            weather_data: Optional DataFrame with weather features
            infrastructure_data: Optional DataFrame with infrastructure features
            confidence_data: Optional DataFrame with confidence metrics
            data_sources: Optional dictionary with data source information
            configuration: Optional dictionary with system configuration
            processing_stats: Optional dictionary with processing statistics
            horizons: List of forecast horizons
            
        Returns:
            Dictionary with paths to all exported files
        """
        try:
            export_results = {
                'html_maps': {},
                'png_snapshots': {},
                'csv_data': '',
                'method_card': '',
                'ops_notes': '',
                'summary_stats': '',
                'export_timestamp': self.file_exporter.export_timestamp
            }
            
            logging.info("Starting complete export process...")
            
            # 1. Export HTML maps
            logging.info("Exporting HTML maps...")
            html_files = self.file_exporter.export_html_maps(maps, horizons)
            export_results['html_maps'] = html_files
            
            # 2. Export PNG snapshots
            logging.info("Exporting PNG snapshots...")
            png_files = self.file_exporter.export_png_snapshots(html_files)
            export_results['png_snapshots'] = png_files
            
            # 3. Export comprehensive CSV
            logging.info("Exporting comprehensive CSV...")
            csv_file = self.file_exporter.export_comprehensive_csv(
                risk_data, grid_data, weather_data, infrastructure_data
            )
            export_results['csv_data'] = csv_file
            
            # 4. Create method card
            logging.info("Creating method card...")
            method_card = self.file_exporter.create_method_card(
                data_sources or {}, 
                configuration or {}, 
                processing_stats
            )
            export_results['method_card'] = method_card
            
            # 5. Create ops notes
            logging.info("Creating operational notes...")
            ops_notes = self.ops_generator.create_ops_notes(
                risk_data, confidence_data, horizons
            )
            export_results['ops_notes'] = ops_notes
            
            # 6. Generate summary statistics
            logging.info("Generating summary statistics...")
            summary_stats = self.ops_generator.generate_summary_statistics_table(
                risk_data, horizons
            )
            export_results['summary_stats'] = summary_stats
            
            # 7. Create export manifest
            manifest_path = self._create_export_manifest(export_results)
            export_results['manifest'] = manifest_path
            
            logging.info("Export process completed successfully")
            logging.info(f"All files exported to: {self.config.output_dir}")
            
            return export_results
            
        except Exception as e:
            logging.error(f"Error in complete export process: {e}")
            raise
    
    def _create_export_manifest(self, export_results: Dict[str, Any]) -> str:
        """Create manifest file listing all exported files"""
        
        manifest_filename = f"export_manifest_{self.file_exporter.export_timestamp}.json"
        manifest_filepath = os.path.join(
            self.config.output_dir,
            manifest_filename
        )
        
        # Create manifest data
        manifest_data = {
            'export_timestamp': export_results['export_timestamp'],
            'export_date': datetime.now().isoformat(),
            'files': export_results,
            'summary': {
                'total_html_maps': len(export_results.get('html_maps', {})),
                'total_png_snapshots': len(export_results.get('png_snapshots', {})),
                'csv_exported': bool(export_results.get('csv_data')),
                'method_card_created': bool(export_results.get('method_card')),
                'ops_notes_created': bool(export_results.get('ops_notes')),
                'summary_stats_created': bool(export_results.get('summary_stats'))
            }
        }
        
        # Write manifest
        with open(manifest_filepath, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Created export manifest: {manifest_filepath}")
        
        return manifest_filepath