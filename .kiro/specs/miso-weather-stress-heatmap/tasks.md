# Implementation Plan

- [x] 1. Set up notebook structure and configuration system
  - Create Jupyter notebook with header, configuration cells, and import statements
  - Implement configuration manager with YAML-based weights and thresholds
  - Set up runtime mode switching (demo/live) and random seed for reproducibility
  - Create directory structure for data/raw/, data/processed/, and output/
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 2. Implement MISO footprint and hexagonal grid generation
  - Create function to define MISO territory boundary from state list or shapefile
  - Implement hexagonal grid generator with ~40km spacing clipped to MISO footprint
  - Add grid cell ID assignment and centroid calculation
  - Write unit tests for spatial accuracy and coverage validation
  - _Requirements: 1.2, 7.4_

- [x] 3. Build weather data ingestion system
  - [x] 3.1 Implement NOAA/NWS gridpoint forecast adapter
    - Create WeatherAdapter base class with standardized interface
    - Implement NOAA API client with error handling and rate limiting
    - Add spatial aggregation from forecast grid to hex cells (mean/max)
    - Write data caching system for raw API responses
    - _Requirements: 2.1, 2.3, 2.4, 2.5_

  - [x] 3.2 Implement Open-Meteo fallback adapter
    - Create Open-Meteo API client following WeatherAdapter interface
    - Implement same spatial aggregation and caching patterns
    - Add automatic fallback logic when NOAA/NWS fails
    - Write tests for adapter switching and data consistency
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [x] 3.3 Create weather feature extraction pipeline
    - Extract temperature, heat index, wind speed, wind gust, precipitation, snow, dewpoint, RH
    - Implement storm probability calculation from precipitation and wind data
    - Add data validation and quality checks for weather parameters
    - Create standardized output format with cell_id, horizon_h, timestamp columns
    - _Requirements: 2.3, 2.4_

- [x] 4. Build infrastructure and exposure data system
  - [x] 4.1 Implement generation capacity data processor
    - Load EIA-860/923 capacity data or equivalent public sources
    - Calculate renewable share (wind+solar) vs total capacity within 50km of each cell
    - Implement spatial joining of capacity points to hex grid cells
    - Add fuel mix breakdown and capacity density calculations
    - _Requirements: 3.2, 3.4_

  - [x] 4.2 Implement population and load exposure processor
    - Load Census population density data at county or tract level
    - Spatially aggregate population data to hex grid cells
    - Add optional load factor weighting for major load centers
    - Normalize population density to [0,1] scale with documented thresholds
    - _Requirements: 3.1, 3.4_

  - [x] 4.3 Create transmission density processor
    - Implement transmission line density calculation if public data available
    - Add fallback to baseline transmission scarcity value (0.5) with documentation
    - Calculate distance to nearest high-voltage line or line crossing count per cell
    - Normalize transmission metrics and handle missing data gracefully
    - _Requirements: 3.3, 3.5_

- [x] 5. Implement feature engineering and normalization system
  - [x] 5.1 Create thermal stress calculation functions
    - Implement heat stress scoring: 0 at HI≤85°F, 1 at HI≥100°F, linear interpolation
    - Implement cold stress scoring: 0 at T≥10°F, 1 at T≤0°F, linear interpolation
    - Combine heat and cold into single thermal stress score (max of both)
    - Write unit tests validating threshold behavior and edge cases
    - _Requirements: 4.1, 4.4_

  - [x] 5.2 Create wind stress calculation functions
    - Implement wind gust scoring: 0 at ≤20mph, 1 at ≥50mph, linear interpolation
    - Add sustained wind bonus: +0.2 if sustained wind ≥30mph
    - Cap maximum wind stress at 1.0 and validate scoring logic
    - Write tests for various wind speed combinations
    - _Requirements: 4.2, 4.4_

  - [x] 5.3 Create precipitation stress calculation functions
    - Implement rain rate scoring: 0 at 0mm/h, 1 at ≥10mm/h
    - Implement snow rate scoring: 0 at 0cm/h, 1 at ≥5cm/h
    - Set ice accretion to immediate maximum score (1.0) for any ice
    - Combine precipitation types into single precipitation stress score
    - _Requirements: 4.3, 4.4_

  - [x] 5.4 Create storm proxy calculation
    - Implement combined storm scoring: precipitation > 0 AND wind gust ≥35mph = 1.0
    - Add scaled scoring based on precipitation × wind gust product for partial conditions
    - Validate storm detection logic against extreme weather scenarios
    - Write tests for storm proxy accuracy and threshold behavior
    - _Requirements: 4.4_

- [x] 6. Build risk scoring engine
  - [x] 6.1 Implement hazard score calculation
    - Combine thermal, wind, precipitation, and storm scores using configurable weights
    - Apply weighted sum: wT×Thermal + wW×Wind + wP×Precip + wS×Storm
    - Normalize hazard scores and validate weight configuration loading
    - Write tests for weight sensitivity and score range validation
    - _Requirements: 4.4, 5.4_

  - [x] 6.2 Implement exposure score calculation
    - Combine population density and optional load factor using configurable weights
    - Apply weighted sum: wPop×PopDensity + wLoad×LoadFactor
    - Handle missing load factor data gracefully with population-only scoring
    - Validate exposure scoring against urban vs rural areas
    - _Requirements: 4.4, 5.4_

  - [x] 6.3 Implement vulnerability score calculation
    - Combine renewable share, transmission scarcity, and outage flags using weights
    - Apply weighted sum: wRen×RenewShare + wTx×TxScarcity + wOut×OutageFlag
    - Handle missing transmission data with documented baseline values
    - Write tests for vulnerability scoring edge cases
    - _Requirements: 4.4, 5.4_

  - [x] 6.4 Create final risk score calculation
    - Implement final risk formula: Risk = zscore(α×H + β×E + γ×V)
    - Calculate z-scores across all cells for each forecast horizon
    - Add configurable blend weights (alpha, beta, gamma) from YAML configuration
    - Validate risk score distribution and mathematical correctness
    - _Requirements: 4.4, 5.4_

- [x] 7. Implement confidence assessment system
  - Create confidence calculation based on data coverage and forecast horizon
  - Implement decreasing confidence with longer forecast horizons
  - Add data quality penalties for missing infrastructure or weather data
  - Calculate per-cell confidence scores and validate confidence ranges [0,1]
  - _Requirements: 6.1, 6.5_

- [x] 8. Build interactive visualization system
  - [x] 8.1 Create Folium map generator
    - Implement choropleth mapping of risk scores with YlOrRd color scale
    - Add layer control for toggling between 12h, 24h, 36h, 48h forecast horizons
    - Create interactive tooltips showing risk score, top 3 contributors, confidence, weather inputs
    - Add map legends, scale bars, and clear captions
    - _Requirements: 1.1, 1.3, 8.3_

  - [x] 8.2 Create summary visualization components
    - Implement top-10 highest risk cells table with contributing factors per horizon
    - Create Plotly bar charts showing risk component breakdown for selected cells
    - Add line charts showing risk evolution across forecast horizons for selected cells
    - Generate ablation analysis charts showing risk sensitivity to component removal
    - _Requirements: 8.1, 8.2_

- [x] 9. Implement validation and quality assurance system
  - [x] 9.1 Create data coverage validation
    - Implement minimum coverage threshold checks for weather and infrastructure data
    - Add automatic degradation to demo mode when coverage falls below thresholds
    - Create user warnings for data gaps and their impact on confidence
    - Log all data quality issues and coverage statistics
    - _Requirements: 6.2, 6.4_

  - [x] 9.2 Implement ablation analysis
    - Create functions to recompute risk with individual components (H, E, V) removed
    - Generate sensitivity analysis showing risk changes when components are dropped
    - Add ablation visualization showing component importance across the grid
    - Validate ablation results against expected component contributions
    - _Requirements: 6.3, 8.2_

- [x] 10. Create output and export system
  - [x] 10.1 Implement standardized file exports
    - Save interactive HTML maps for each forecast horizon (12h, 24h, 36h, 48h)
    - Generate PNG snapshots of all maps for static reporting
    - Export comprehensive CSV with cell_id × horizon_h × risk scores × components
    - Create method card (markdown) documenting data sources, methodology, and limitations
    - _Requirements: 1.4, 5.5_

  - [x] 10.2 Generate operational summary outputs
    - Create "Ops Notes" text file with top hotspots, risk drivers, and confidence assessments
    - Generate summary statistics table showing risk distribution by horizon
    - Add data freshness timestamps and API source documentation
    - Include clear disclaimers about assumptions, proxy usage, and data limitations
    - _Requirements: 8.1, 8.4, 8.5_

- [ ] 11. Add comprehensive logging and documentation
  - Implement comprehensive logging of all configuration parameters and data source URLs
  - Add execution timing and performance monitoring for large spatial operations
  - Create inline markdown documentation explaining each processing step
  - Generate reproducibility report with random seeds, versions, and data timestamps
  - _Requirements: 6.4, 7.3_

- [ ] 12. Create extensibility framework
  - [ ] 12.1 Design plugin architecture for additional stressors
    - Create stub functions for resource transition indicators and load growth factors
    - Implement standardized interface for new risk components
    - Add configuration hooks for additional weights and parameters
    - Document extension patterns and provide example implementations
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 12.2 Add demo data and testing framework
    - Create sample datasets for demo mode operation
    - Implement unit tests for all core mathematical functions
    - Add integration tests for complete pipeline execution
    - Create performance benchmarks and validation against known scenarios
    - _Requirements: 5.1, 7.4_