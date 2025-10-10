"""
Test suite for Export System

Tests the comprehensive export functionality including standardized file exports,
operational summary generation, and complete export workflows.

Requirements tested:
- 1.4: Export HTML maps, PNG snapshots, and CSV data files
- 5.5: Method card documenting data sources, methodology, and limitations
- 8.1: Ops Notes with top hotspots and risk drivers
- 8.4: Data freshness timestamps and API source documentation
- 8.5: Clear disclaimers about assumptions and limitations
"""

import pytest
import os
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Polygon
import numpy as np

from export_system import (
    ExportConfig, StandardizedFileExporter, OperationalSummaryGenerator,
    ExportSystem
)


class TestExportConfig:
    """Test export configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ExportConfig()
        
        assert config.output_dir == "output"
        assert config.maps_dir == "maps"
        assert config.data_dir == "data"
        assert config.docs_dir == "docs"
        assert config.png_width == 1200
        assert config.png_height == 800
        assert config.csv_precision == 6
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ExportConfig(
            output_dir="custom_output",
            png_width=1600,
            csv_precision=4
        )
        
        assert config.output_dir == "custom_output"
        assert config.png_width == 1600
        assert config.csv_precision == 4


class TestStandardizedFileExporter:
    """Test standardized file export functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return ExportConfig(output_dir=temp_dir)
    
    @pytest.fixture
    def exporter(self, config):
        """Create file exporter instance"""
        return StandardizedFileExporter(config)
    
    @pytest.fixture
    def sample_maps(self):
        """Create sample Folium maps"""
        maps = {}
        for horizon in [12, 24, 36, 48]:
            map_obj = folium.Map(location=[40.0, -89.0], zoom_start=6)
            folium.Marker([40.0, -89.0], popup=f"{horizon}h forecast").add_to(map_obj)
            maps[horizon] = map_obj
        return maps
    
    @pytest.fixture
    def sample_risk_data(self):
        """Create sample risk data"""
        np.random.seed(42)
        
        data = []
        for horizon in [12, 24, 36, 48]:
            for cell_id in range(100):
                data.append({
                    'cell_id': f'cell_{cell_id:03d}',
                    'horizon_h': horizon,
                    'final_risk': np.random.normal(0, 1),
                    'hazard_score': np.random.uniform(0, 1),
                    'exposure_score': np.random.uniform(0, 1),
                    'vulnerability_score': np.random.uniform(0, 1),
                    'confidence': np.random.uniform(0.5, 1.0),
                    'thermal_stress': np.random.uniform(0, 1),
                    'wind_stress': np.random.uniform(0, 1),
                    'precip_stress': np.random.uniform(0, 1),
                    'storm_proxy': np.random.uniform(0, 1),
                    'temp_2m': np.random.uniform(20, 100),
                    'heat_index': np.random.uniform(20, 110),
                    'wind_speed': np.random.uniform(0, 40),
                    'wind_gust': np.random.uniform(0, 60),
                    'precip_rate': np.random.uniform(0, 20)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_grid_data(self):
        """Create sample grid data"""
        data = []
        for cell_id in range(100):
            # Create simple square polygons
            x = (cell_id % 10) * 0.1
            y = (cell_id // 10) * 0.1
            polygon = Polygon([
                (x, y), (x + 0.1, y), (x + 0.1, y + 0.1), (x, y + 0.1)
            ])
            
            data.append({
                'cell_id': f'cell_{cell_id:03d}',
                'geometry': polygon,
                'centroid_lat': y + 0.05,
                'centroid_lon': x + 0.05,
                'area_km2': 1600.0  # ~40km x 40km
            })
        
        return gpd.GeoDataFrame(data)
    
    def test_initialization(self, exporter, temp_dir):
        """Test exporter initialization"""
        assert exporter.config.output_dir == temp_dir
        assert os.path.exists(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "maps"))
        assert os.path.exists(os.path.join(temp_dir, "data"))
        assert os.path.exists(os.path.join(temp_dir, "docs"))
    
    def test_export_html_maps(self, exporter, sample_maps):
        """Test HTML map export"""
        exported_files = exporter.export_html_maps(sample_maps)
        
        # Check that files were created for each horizon
        assert len(exported_files) == 5  # 4 horizons + 1 combined
        
        for horizon in [12, 24, 36, 48]:
            assert horizon in exported_files
            assert os.path.exists(exported_files[horizon])
            assert exported_files[horizon].endswith('.html')
        
        # Check combined map
        assert 'combined' in exported_files
        assert os.path.exists(exported_files['combined'])
    
    def test_export_comprehensive_csv(self, exporter, sample_risk_data, sample_grid_data):
        """Test comprehensive CSV export"""
        # Create sample weather and infrastructure data
        weather_data = sample_risk_data[['cell_id', 'horizon_h', 'temp_2m', 'heat_index', 
                                       'wind_speed', 'wind_gust', 'precip_rate']].copy()
        weather_data['snow_rate'] = 0.0
        
        infrastructure_data = pd.DataFrame({
            'cell_id': [f'cell_{i:03d}' for i in range(100)],
            'population_density': np.random.uniform(0, 1, 100),
            'renewable_share': np.random.uniform(0, 1, 100),
            'transmission_density': np.random.uniform(0, 1, 100)
        })
        
        csv_path = exporter.export_comprehensive_csv(
            sample_risk_data, sample_grid_data, weather_data, infrastructure_data
        )
        
        # Check that file was created
        assert os.path.exists(csv_path)
        assert csv_path.endswith('.csv')
        
        # Check CSV content
        exported_df = pd.read_csv(csv_path)
        assert len(exported_df) > 0
        assert 'cell_id' in exported_df.columns
        assert 'horizon_h' in exported_df.columns
        assert 'final_risk' in exported_df.columns
        assert 'centroid_lat' in exported_df.columns
        # Weather data should be merged (temp_2m already exists in risk_data, so won't be duplicated)
        assert any('temp_2m' in col for col in exported_df.columns)
        assert 'population_density' in exported_df.columns
    
    def test_create_method_card(self, exporter):
        """Test method card creation"""
        data_sources = {
            'weather': {
                'primary_source': 'NOAA/NWS',
                'fallback_source': 'Open-Meteo',
                'spatial_resolution': '3km grid',
                'temporal_resolution': 'Hourly'
            },
            'infrastructure': {
                'capacity_source': 'EIA-860',
                'population_source': 'US Census',
                'transmission_source': 'Public data'
            }
        }
        
        configuration = {
            'weights': {
                'hazard': {'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15},
                'exposure': {'pop': 0.7, 'load': 0.3},
                'vulnerability': {'renew_share': 0.6, 'tx_scarcity': 0.3},
                'blend': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
            },
            'thresholds': {
                'thermal': {'heat_low': 85, 'heat_high': 100, 'cold_low': 10, 'cold_high': 0},
                'wind': {'gust_low': 20, 'gust_high': 50, 'sustained_threshold': 30},
                'precip': {'rain_heavy': 10, 'snow_heavy': 5}
            },
            'runtime': {'random_seed': 42}
        }
        
        processing_stats = {
            'processing_date': '2024-01-01',
            'total_cells': 1000,
            'weather_coverage': 95.5,
            'infrastructure_coverage': 88.2,
            'processing_time': 120.5
        }
        
        method_card_path = exporter.create_method_card(
            data_sources, configuration, processing_stats
        )
        
        # Check that file was created
        assert os.path.exists(method_card_path)
        assert method_card_path.endswith('.md')
        
        # Check content
        with open(method_card_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'MISO Weather-Stress Heatmap - Method Card' in content
        assert 'Risk = zscore(α × Hazard + β × Exposure + γ × Vulnerability)' in content
        assert 'NOAA/NWS' in content
        assert 'Thermal: 0.3' in content
        assert '**Total Cells:** 1000' in content
        # Check for key sections that should be present
        assert ('Limitations and Assumptions' in content or 
                'proxy data and simplified models' in content)
    
    @patch('export_system.SELENIUM_AVAILABLE', False)
    def test_export_png_snapshots_fallback(self, exporter, sample_maps):
        """Test PNG export with matplotlib fallback"""
        # First export HTML maps
        html_files = exporter.export_html_maps(sample_maps)
        
        # Export PNG snapshots (should use matplotlib fallback)
        png_files = exporter.export_png_snapshots(html_files, use_selenium=False)
        
        # Check that PNG files were created
        for horizon in [12, 24, 36, 48]:
            if horizon in png_files:
                assert os.path.exists(png_files[horizon])
                assert png_files[horizon].endswith('.png')


class TestOperationalSummaryGenerator:
    """Test operational summary generation"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return ExportConfig(output_dir=temp_dir)
    
    @pytest.fixture
    def ops_generator(self, config):
        """Create ops generator instance"""
        return OperationalSummaryGenerator(config)
    
    @pytest.fixture
    def sample_risk_data(self):
        """Create sample risk data with realistic patterns"""
        np.random.seed(42)
        
        data = []
        for horizon in [12, 24, 36, 48]:
            for cell_id in range(50):
                # Create some high-risk cells
                if cell_id < 5:
                    risk_score = np.random.uniform(2.0, 3.0)  # High risk
                    hazard = np.random.uniform(0.7, 1.0)
                    exposure = np.random.uniform(0.6, 0.9)
                    vulnerability = np.random.uniform(0.5, 0.8)
                else:
                    risk_score = np.random.normal(0, 1)  # Normal distribution
                    hazard = np.random.uniform(0, 0.7)
                    exposure = np.random.uniform(0, 0.6)
                    vulnerability = np.random.uniform(0, 0.5)
                
                data.append({
                    'cell_id': f'cell_{cell_id:03d}',
                    'horizon_h': horizon,
                    'final_risk': risk_score,
                    'hazard_score': hazard,
                    'exposure_score': exposure,
                    'vulnerability_score': vulnerability,
                    'thermal_stress': np.random.uniform(0, 1),
                    'wind_stress': np.random.uniform(0, 1),
                    'precip_stress': np.random.uniform(0, 1),
                    'storm_proxy': np.random.uniform(0, 1),
                    'temp_2m': np.random.uniform(20, 100),
                    'heat_index': np.random.uniform(20, 110),
                    'wind_gust': np.random.uniform(0, 60),
                    'precip_rate': np.random.uniform(0, 20)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_confidence_data(self):
        """Create sample confidence data"""
        np.random.seed(42)
        
        data = []
        for horizon in [12, 24, 36, 48]:
            # Confidence decreases with horizon
            base_confidence = 0.9 - (horizon - 12) * 0.05
            
            for cell_id in range(50):
                confidence = np.random.uniform(
                    max(0.5, base_confidence - 0.2),
                    min(1.0, base_confidence + 0.1)
                )
                
                data.append({
                    'cell_id': f'cell_{cell_id:03d}',
                    'horizon_h': horizon,
                    'confidence': confidence
                })
        
        return pd.DataFrame(data)
    
    def test_create_ops_notes(self, ops_generator, sample_risk_data, sample_confidence_data):
        """Test ops notes creation"""
        ops_notes_path = ops_generator.create_ops_notes(
            sample_risk_data, sample_confidence_data
        )
        
        # Check that file was created
        assert os.path.exists(ops_notes_path)
        assert ops_notes_path.endswith('.txt')
        
        # Check content
        with open(ops_notes_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'MISO WEATHER-STRESS HEATMAP - OPERATIONAL NOTES' in content
        assert 'TOP RISK HOTSPOTS BY FORECAST HORIZON' in content
        assert '12h FORECAST - TOP' in content
        assert 'RISK DISTRIBUTION SUMMARY' in content
        assert 'CONFIDENCE ASSESSMENT' in content
        assert 'DATA SOURCES AND FRESHNESS' in content
        assert 'DISCLAIMERS AND LIMITATIONS' in content
        
        # Check that high-risk cells are identified
        assert 'cell_000' in content or 'cell_001' in content  # Should be in top hotspots
    
    def test_generate_summary_statistics_table(self, ops_generator, sample_risk_data):
        """Test summary statistics table generation"""
        summary_path = ops_generator.generate_summary_statistics_table(sample_risk_data)
        
        # Check that file was created
        assert os.path.exists(summary_path)
        assert summary_path.endswith('.csv')
        
        # Check content
        summary_df = pd.read_csv(summary_path)
        
        assert len(summary_df) == 4  # One row per horizon
        assert 'horizon_h' in summary_df.columns
        assert 'total_cells' in summary_df.columns
        assert 'mean_risk' in summary_df.columns
        assert 'std_risk' in summary_df.columns
        assert 'p95_risk' in summary_df.columns
        assert 'high_risk_cells' in summary_df.columns
        
        # Check that all horizons are present
        assert set(summary_df['horizon_h']) == {12, 24, 36, 48}
    
    def test_get_primary_risk_driver(self, ops_generator):
        """Test primary risk driver identification"""
        # Test hazard-dominated case
        row = pd.Series({
            'hazard_score': 0.8,
            'exposure_score': 0.3,
            'vulnerability_score': 0.2,
            'thermal_stress': 0.9,
            'wind_stress': 0.1,
            'precip_stress': 0.1,
            'storm_proxy': 0.1
        })
        
        driver = ops_generator._get_primary_risk_driver(row)
        assert 'Hazard (Weather)' in driver
        assert 'Thermal Stress' in driver
        
        # Test exposure-dominated case
        row = pd.Series({
            'hazard_score': 0.2,
            'exposure_score': 0.8,
            'vulnerability_score': 0.3
        })
        
        driver = ops_generator._get_primary_risk_driver(row)
        assert driver == 'Exposure (Population)'
    
    def test_get_weather_summary(self, ops_generator):
        """Test weather summary generation"""
        row = pd.Series({
            'temp_2m': 95.5,
            'heat_index': 105.2,
            'wind_gust': 45.3,
            'precip_rate': 8.7
        })
        
        summary = ops_generator._get_weather_summary(row)
        
        assert 'T=95.5°F' in summary
        assert 'HI=105.2°F' in summary
        assert 'Gust=45.3mph' in summary
        assert 'Precip=8.7mm/h' in summary


class TestExportSystem:
    """Test complete export system integration"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return ExportConfig(output_dir=temp_dir)
    
    @pytest.fixture
    def export_system(self, config):
        """Create export system instance"""
        return ExportSystem(config)
    
    @pytest.fixture
    def complete_test_data(self):
        """Create complete test dataset"""
        np.random.seed(42)
        
        # Risk data
        risk_data = []
        for horizon in [12, 24, 36, 48]:
            for cell_id in range(20):
                risk_data.append({
                    'cell_id': f'cell_{cell_id:03d}',
                    'horizon_h': horizon,
                    'final_risk': np.random.normal(0, 1),
                    'hazard_score': np.random.uniform(0, 1),
                    'exposure_score': np.random.uniform(0, 1),
                    'vulnerability_score': np.random.uniform(0, 1),
                    'thermal_stress': np.random.uniform(0, 1),
                    'wind_stress': np.random.uniform(0, 1),
                    'precip_stress': np.random.uniform(0, 1),
                    'storm_proxy': np.random.uniform(0, 1),
                    'temp_2m': np.random.uniform(20, 100),
                    'heat_index': np.random.uniform(20, 110),
                    'wind_speed': np.random.uniform(0, 40),
                    'wind_gust': np.random.uniform(0, 60),
                    'precip_rate': np.random.uniform(0, 20)
                })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Grid data
        grid_data = []
        for cell_id in range(20):
            x = (cell_id % 5) * 0.1
            y = (cell_id // 5) * 0.1
            polygon = Polygon([
                (x, y), (x + 0.1, y), (x + 0.1, y + 0.1), (x, y + 0.1)
            ])
            
            grid_data.append({
                'cell_id': f'cell_{cell_id:03d}',
                'geometry': polygon,
                'centroid_lat': y + 0.05,
                'centroid_lon': x + 0.05,
                'area_km2': 1600.0
            })
        
        grid_gdf = gpd.GeoDataFrame(grid_data)
        
        # Maps
        maps = {}
        for horizon in [12, 24, 36, 48]:
            map_obj = folium.Map(location=[40.0, -89.0], zoom_start=6)
            folium.Marker([40.0, -89.0], popup=f"{horizon}h forecast").add_to(map_obj)
            maps[horizon] = map_obj
        
        # Confidence data
        confidence_data = []
        for horizon in [12, 24, 36, 48]:
            for cell_id in range(20):
                confidence_data.append({
                    'cell_id': f'cell_{cell_id:03d}',
                    'horizon_h': horizon,
                    'confidence': np.random.uniform(0.6, 1.0)
                })
        
        confidence_df = pd.DataFrame(confidence_data)
        
        # Data sources and configuration
        data_sources = {
            'weather': {
                'primary_source': 'NOAA/NWS',
                'fallback_source': 'Open-Meteo'
            },
            'infrastructure': {
                'capacity_source': 'EIA-860',
                'population_source': 'US Census'
            }
        }
        
        configuration = {
            'weights': {
                'hazard': {'thermal': 0.3, 'wind': 0.3, 'precip': 0.25, 'storm': 0.15},
                'blend': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
            },
            'runtime': {'random_seed': 42}
        }
        
        return {
            'maps': maps,
            'risk_data': risk_df,
            'grid_data': grid_gdf,
            'confidence_data': confidence_df,
            'data_sources': data_sources,
            'configuration': configuration
        }
    
    @patch('export_system.SELENIUM_AVAILABLE', False)  # Use matplotlib fallback
    def test_export_all_outputs(self, export_system, complete_test_data):
        """Test complete export workflow"""
        results = export_system.export_all_outputs(
            maps=complete_test_data['maps'],
            risk_data=complete_test_data['risk_data'],
            grid_data=complete_test_data['grid_data'],
            confidence_data=complete_test_data['confidence_data'],
            data_sources=complete_test_data['data_sources'],
            configuration=complete_test_data['configuration']
        )
        
        # Check that all expected outputs were created
        assert 'html_maps' in results
        assert 'png_snapshots' in results
        assert 'csv_data' in results
        assert 'method_card' in results
        assert 'ops_notes' in results
        assert 'summary_stats' in results
        assert 'manifest' in results
        
        # Check HTML maps
        assert len(results['html_maps']) == 5  # 4 horizons + combined
        for horizon in [12, 24, 36, 48]:
            assert horizon in results['html_maps']
            assert os.path.exists(results['html_maps'][horizon])
        
        # Check PNG snapshots
        for horizon in [12, 24, 36, 48]:
            if horizon in results['png_snapshots']:
                assert os.path.exists(results['png_snapshots'][horizon])
        
        # Check CSV data
        assert os.path.exists(results['csv_data'])
        csv_df = pd.read_csv(results['csv_data'])
        assert len(csv_df) > 0
        assert 'cell_id' in csv_df.columns
        
        # Check method card
        assert os.path.exists(results['method_card'])
        with open(results['method_card'], 'r') as f:
            method_content = f.read()
        assert 'Method Card' in method_content
        
        # Check ops notes
        assert os.path.exists(results['ops_notes'])
        with open(results['ops_notes'], 'r') as f:
            ops_content = f.read()
        assert 'OPERATIONAL NOTES' in ops_content
        
        # Check summary stats
        assert os.path.exists(results['summary_stats'])
        stats_df = pd.read_csv(results['summary_stats'])
        assert len(stats_df) == 4  # One row per horizon
        
        # Check manifest
        assert os.path.exists(results['manifest'])
        with open(results['manifest'], 'r') as f:
            manifest = json.load(f)
        assert 'export_timestamp' in manifest
        assert 'files' in manifest
        assert 'summary' in manifest
    
    def test_create_export_manifest(self, export_system):
        """Test export manifest creation"""
        export_results = {
            'html_maps': {12: 'map_12h.html', 24: 'map_24h.html'},
            'png_snapshots': {12: 'map_12h.png'},
            'csv_data': 'data.csv',
            'method_card': 'method.md',
            'ops_notes': 'ops.txt',
            'summary_stats': 'stats.csv',
            'export_timestamp': '20240101_120000'
        }
        
        manifest_path = export_system._create_export_manifest(export_results)
        
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert manifest['export_timestamp'] == '20240101_120000'
        assert manifest['summary']['total_html_maps'] == 2
        assert manifest['summary']['total_png_snapshots'] == 1
        assert manifest['summary']['csv_exported'] is True


if __name__ == "__main__":
    pytest.main([__file__])