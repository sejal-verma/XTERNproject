"""
Test Suite for Visualization System

This module contains comprehensive tests for the interactive visualization system,
including map generation, summary visualizations, and ablation analysis.

Tests cover:
- Folium map generation with choropleth and layer controls
- Interactive tooltips and legends
- Summary tables and charts
- Ablation analysis functionality
- Export and validation functions
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
import plotly.graph_objects as go
import tempfile
import os
import logging

# Import modules to test
from visualization_system import (
    VisualizationSystem, 
    FoliumMapGenerator, 
    SummaryVisualizationGenerator,
    AblationAnalysisEngine,
    VisualizationConfig,
    VisualizationValidation,
    VisualizationTests
)
from risk_scoring_engine import RiskScoringEngine


class TestVisualizationSystem:
    """Test suite for the main visualization system"""
    
    @pytest.fixture
    def sample_grid_data(self):
        """Create sample hexagonal grid data for testing"""
        
        # Create simple hexagonal polygons
        hex_coords = [
            [(-90, 40), (-89.5, 40.5), (-89, 40), (-89.5, 39.5), (-90, 40)],
            [(-89, 40), (-88.5, 40.5), (-88, 40), (-88.5, 39.5), (-89, 40)],
            [(-88, 40), (-87.5, 40.5), (-87, 40), (-87.5, 39.5), (-88, 40)]
        ]
        
        geometries = [Polygon(coords) for coords in hex_coords]
        
        grid_data = gpd.GeoDataFrame({
            'cell_id': ['hex_0001', 'hex_0002', 'hex_0003'],
            'centroid_lon': [-89.5, -88.5, -87.5],
            'centroid_lat': [40.0, 40.0, 40.0],
            'area_km2': [1600, 1600, 1600],
            'geometry': geometries
        }, crs='EPSG:4326')
        
        return grid_data
    
    @pytest.fixture
    def sample_risk_data(self):
        """Create sample risk assessment data for testing"""
        
        # Create data for multiple horizons and cells
        data = []
        cell_ids = ['hex_0001', 'hex_0002', 'hex_0003']
        horizons = [12, 24, 36, 48]
        
        np.random.seed(42)  # Reproducible test data
        
        for horizon in horizons:
            for i, cell_id in enumerate(cell_ids):
                # Generate realistic risk scores
                base_risk = np.random.normal(0, 1)  # Z-scored risk
                
                data.append({
                    'cell_id': cell_id,
                    'horizon_h': horizon,
                    'final_risk': base_risk,
                    'hazard_score': np.random.uniform(0, 1),
                    'exposure_score': np.random.uniform(0, 1),
                    'vulnerability_score': np.random.uniform(0, 1),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'thermal_stress': np.random.uniform(0, 1),
                    'wind_stress': np.random.uniform(0, 1),
                    'precip_stress': np.random.uniform(0, 1),
                    'storm_proxy': np.random.uniform(0, 1),
                    'normalized_pop_density': np.random.uniform(0, 1),
                    'renewable_share': np.random.uniform(0, 1),
                    'transmission_scarcity': np.random.uniform(0, 1),
                    'temp_2m': np.random.uniform(20, 100),
                    'heat_index': np.random.uniform(20, 110),
                    'wind_speed': np.random.uniform(0, 40),
                    'wind_gust': np.random.uniform(0, 60),
                    'precip_rate': np.random.uniform(0, 15)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def viz_system(self):
        """Create visualization system instance for testing"""
        config = VisualizationConfig()
        return VisualizationSystem(config)
    
    @pytest.fixture
    def risk_engine(self):
        """Create risk scoring engine for ablation tests"""
        return RiskScoringEngine()
    
    def test_visualization_system_initialization(self, viz_system):
        """Test visualization system initializes correctly"""
        
        assert isinstance(viz_system, VisualizationSystem)
        assert isinstance(viz_system.map_generator, FoliumMapGenerator)
        assert isinstance(viz_system.summary_generator, SummaryVisualizationGenerator)
        assert isinstance(viz_system.config, VisualizationConfig)
    
    def test_folium_map_generation(self, viz_system, sample_grid_data, sample_risk_data):
        """Test Folium map generation with choropleth and layers"""
        
        # Generate map
        risk_map = viz_system.map_generator.create_risk_heatmap(
            sample_grid_data, sample_risk_data, [12, 24]
        )
        
        # Validate map
        assert isinstance(risk_map, folium.Map)
        
        # Check for layer control
        has_layer_control = any(
            isinstance(child, folium.LayerControl) 
            for child in risk_map._children.values()
        )
        assert has_layer_control
        
        # Check for feature groups (layers)
        feature_groups = [
            child for child in risk_map._children.values() 
            if isinstance(child, folium.FeatureGroup)
        ]
        assert len(feature_groups) >= 1  # At least one layer should be created
    
    def test_tooltip_generation(self, viz_system, sample_risk_data):
        """Test interactive tooltip content generation"""
        
        # Get sample row
        sample_row = sample_risk_data.iloc[0]
        
        # Generate tooltip
        tooltip_html = viz_system.map_generator._create_tooltip_html(sample_row, 12)
        
        # Check tooltip contains required elements
        required_elements = [
            'Risk Score', 'Confidence', 'Risk Components', 
            'Top Contributors', 'Weather Inputs'
        ]
        
        for element in required_elements:
            assert element in tooltip_html
        
        # Check HTML structure
        assert '<div' in tooltip_html
        assert '<table' in tooltip_html
        assert tooltip_html.count('<tr>') >= 3  # Multiple table rows
    
    def test_color_scaling(self, viz_system):
        """Test risk score color scaling"""
        
        # Test color generation for different risk levels
        test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        colors = [
            viz_system.map_generator._get_color_from_risk(val) 
            for val in test_values
        ]
        
        # Check all colors are valid hex codes
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith('#')
            assert len(color) == 7
        
        # Check color progression (should be different)
        assert len(set(colors)) > 1
    
    def test_top_risk_table_generation(self, viz_system, sample_risk_data):
        """Test top risk cells table generation"""
        
        # Generate table
        top_table = viz_system.summary_generator.create_top_risk_table(
            sample_risk_data, 12, n_top=3
        )
        
        # Validate table structure
        assert isinstance(top_table, pd.DataFrame)
        assert len(top_table) <= 3  # Should not exceed n_top
        
        expected_columns = [
            'Rank', 'Cell ID', 'Risk Score', 'Hazard', 
            'Exposure', 'Vulnerability', 'Confidence', 'Top Contributor'
        ]
        
        for col in expected_columns:
            assert col in top_table.columns
        
        # Check data types and ranges
        assert top_table['Rank'].dtype in [int, 'int64']
        assert top_table['Risk Score'].between(-5, 5).all()  # Reasonable z-score range
        assert top_table['Confidence'].between(0, 100).all()  # Percentage
    
    def test_risk_component_chart(self, viz_system, sample_risk_data):
        """Test risk component breakdown chart generation"""
        
        # Get sample cell IDs
        sample_cells = sample_risk_data['cell_id'].unique()[:2]
        
        # Generate chart
        component_chart = viz_system.summary_generator.create_risk_component_chart(
            sample_risk_data, sample_cells, 12
        )
        
        # Validate chart
        assert isinstance(component_chart, go.Figure)
        assert len(component_chart.data) >= 3  # Should have hazard, exposure, vulnerability
        
        # Check for proper layout
        assert component_chart.layout.title.text is not None
        assert component_chart.layout.xaxis.title.text is not None
        assert component_chart.layout.yaxis.title.text is not None
    
    def test_risk_evolution_chart(self, viz_system, sample_risk_data):
        """Test risk evolution across horizons chart"""
        
        # Get sample cell IDs
        sample_cells = sample_risk_data['cell_id'].unique()[:2]
        
        # Generate chart
        evolution_chart = viz_system.summary_generator.create_risk_evolution_chart(
            sample_risk_data, sample_cells, [12, 24, 36, 48]
        )
        
        # Validate chart
        assert isinstance(evolution_chart, go.Figure)
        assert len(evolution_chart.data) >= 1  # At least one line
        
        # Check for proper layout
        assert 'Evolution' in evolution_chart.layout.title.text
        assert evolution_chart.layout.xaxis.title.text is not None
        assert evolution_chart.layout.yaxis.title.text is not None
    
    def test_ablation_analysis(self, viz_system, sample_risk_data, risk_engine):
        """Test ablation analysis functionality"""
        
        # Initialize ablation engine
        ablation_engine = AblationAnalysisEngine(risk_engine)
        
        # Filter data for single horizon
        horizon_data = sample_risk_data[sample_risk_data['horizon_h'] == 12].copy()
        
        # Perform ablation analysis
        ablation_results = ablation_engine.perform_ablation_analysis(horizon_data)
        
        # Validate results
        assert isinstance(ablation_results, dict)
        assert 'hazard' in ablation_results
        assert 'exposure' in ablation_results
        assert 'vulnerability' in ablation_results
        
        # Check each result is a DataFrame with correct structure
        for component, result_df in ablation_results.items():
            assert isinstance(result_df, pd.DataFrame)
            assert 'cell_id' in result_df.columns
            assert 'final_risk' in result_df.columns
            assert len(result_df) == len(horizon_data)
    
    def test_component_importance_calculation(self, sample_risk_data, risk_engine):
        """Test component importance calculation from ablation results"""
        
        # Initialize ablation engine
        ablation_engine = AblationAnalysisEngine(risk_engine)
        
        # Filter data for single horizon
        horizon_data = sample_risk_data[sample_risk_data['horizon_h'] == 12].copy()
        
        # Perform ablation analysis
        ablation_results = ablation_engine.perform_ablation_analysis(horizon_data)
        
        # Calculate importance
        importance_df = ablation_engine.calculate_component_importance(
            horizon_data, ablation_results
        )
        
        # Validate importance results
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3  # Three components tested
        
        expected_columns = [
            'component', 'mean_impact', 'std_impact', 'max_impact', 
            'min_impact', 'median_impact', 'cells_affected', 'total_cells'
        ]
        
        for col in expected_columns:
            assert col in importance_df.columns
        
        # Check data reasonableness
        assert importance_df['total_cells'].iloc[0] == len(horizon_data)
        assert importance_df['cells_affected'].between(0, len(horizon_data)).all()
    
    def test_map_export_functionality(self, viz_system):
        """Test map export to HTML file"""
        
        # Create simple test map
        test_map = folium.Map(location=[40, -89], zoom_start=6)
        
        # Test export
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            try:
                viz_system.map_generator.export_map(test_map, tmp.name, "Test Map")
                
                # Check file exists and has content
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                # Check file contains expected HTML elements
                with open(tmp.name, 'r') as f:
                    content = f.read()
                    assert '<html>' in content
                    assert 'Test Map' in content
                    
            finally:
                # Clean up
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_complete_visualization_pipeline(self, viz_system, sample_grid_data, sample_risk_data):
        """Test complete visualization generation pipeline"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate all visualizations
            results = viz_system.generate_all_visualizations(
                sample_grid_data, sample_risk_data, tmp_dir, [12, 24]
            )
            
            # Validate results structure
            assert 'maps' in results
            assert 'charts' in results
            assert 'tables' in results
            assert 'summary' in results
            
            # Check main map was created
            assert 'main_heatmap' in results['maps']
            main_map_path = results['maps']['main_heatmap']
            assert os.path.exists(main_map_path)
            assert main_map_path.endswith('.html')
            
            # Check summary statistics were created
            if 'statistics' in results['summary']:
                stats_path = results['summary']['statistics']
                assert os.path.exists(stats_path)
                assert stats_path.endswith('.json')
    
    def test_validation_functions(self, sample_risk_data):
        """Test visualization validation functions"""
        
        # Test map validation
        test_map = folium.Map(location=[40, -89])
        map_validation = VisualizationValidation.validate_map_generation(test_map, 0)
        
        assert isinstance(map_validation, dict)
        assert 'is_folium_map' in map_validation
        assert map_validation['is_folium_map'] is True
        
        # Test chart validation
        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
        test_fig.update_layout(title="Test Chart", xaxis_title="X", yaxis_title="Y")
        
        chart_validation = VisualizationValidation.validate_chart_generation(test_fig)
        
        assert isinstance(chart_validation, dict)
        assert chart_validation['is_plotly_figure'] is True
        assert chart_validation['has_data'] is True
        assert chart_validation['has_title'] is True
        
        # Test table validation
        test_df = pd.DataFrame({
            'Risk Score': [0.5, 1.2, -0.3],
            'Cell ID': ['hex_001', 'hex_002', 'hex_003']
        })
        
        table_validation = VisualizationValidation.validate_table_generation(
            test_df, ['Risk Score', 'Cell ID']
        )
        
        assert isinstance(table_validation, dict)
        assert table_validation['is_dataframe'] is True
        assert table_validation['has_data'] is True
        assert table_validation['has_required_columns'] is True


class TestVisualizationIntegration:
    """Integration tests for visualization system with other components"""
    
    def test_integration_with_risk_engine(self):
        """Test visualization system integration with risk scoring engine"""
        
        # This test would require actual integration with the risk scoring engine
        # For now, we'll test that the interfaces are compatible
        
        risk_engine = RiskScoringEngine()
        viz_system = VisualizationSystem()
        
        # Test that ablation engine can be initialized with risk engine
        ablation_engine = AblationAnalysisEngine(risk_engine)
        
        assert ablation_engine.risk_engine == risk_engine
        assert hasattr(ablation_engine.risk_engine, 'weights')
    
    def test_data_format_compatibility(self):
        """Test that visualization system handles expected data formats correctly"""
        
        # Test with minimal required columns
        minimal_risk_data = pd.DataFrame({
            'cell_id': ['hex_001', 'hex_002'],
            'horizon_h': [12, 12],
            'final_risk': [0.5, -0.3],
            'hazard_score': [0.6, 0.4],
            'exposure_score': [0.7, 0.2],
            'vulnerability_score': [0.3, 0.8],
            'confidence': [0.9, 0.85]
        })
        
        viz_system = VisualizationSystem()
        
        # Test table generation with minimal data
        top_table = viz_system.summary_generator.create_top_risk_table(
            minimal_risk_data, 12, n_top=2
        )
        
        assert len(top_table) == 2
        assert 'Risk Score' in top_table.columns


if __name__ == "__main__":
    # Run tests if script is executed directly
    import sys
    
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test instances
    test_viz = TestVisualizationSystem()
    test_integration = TestVisualizationIntegration()
    
    print("Running visualization system tests...")
    
    try:
        # Run basic functionality tests
        print("\n1. Testing visualization system initialization...")
        viz_system = VisualizationSystem()
        test_viz.test_visualization_system_initialization(viz_system)
        print("✓ Initialization test passed")
        
        print("\n2. Testing color scaling...")
        test_viz.test_color_scaling(viz_system)
        print("✓ Color scaling test passed")
        
        print("\n3. Testing validation functions...")
        sample_risk_data = pd.DataFrame({
            'cell_id': ['hex_001'],
            'horizon_h': [12],
            'final_risk': [0.5]
        })
        test_viz.test_validation_functions(sample_risk_data)
        print("✓ Validation functions test passed")
        
        print("\n4. Testing integration compatibility...")
        test_integration.test_integration_with_risk_engine()
        test_integration.test_data_format_compatibility()
        print("✓ Integration tests passed")
        
        print("\n✓ All basic tests passed successfully!")
        print("\nNote: Full test suite requires pytest and sample data.")
        print("Run 'pytest test_visualization.py -v' for comprehensive testing.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)