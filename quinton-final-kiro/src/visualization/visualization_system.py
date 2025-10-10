"""
Interactive Visualization System for MISO Weather-Stress Heatmap

This module implements the comprehensive visualization system that creates
interactive maps and summary visualizations for grid stress risk assessment.

Key Components:
- Folium map generator with choropleth mapping and layer controls
- Interactive tooltips with risk breakdowns and confidence metrics
- Summary visualization components (tables, charts, ablation analysis)
- Export functionality for HTML maps, PNG snapshots, and CSV data

Requirements addressed:
- 1.1: Interactive heat maps for multiple forecast horizons
- 1.3: Interactive tooltips with risk scores and contributing factors
- 8.1: Top-10 highest risk cells with contributing factors
- 8.2: Risk component breakdown and evolution charts
- 8.3: Clear color scales, legends, and captions
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import geopandas as gpd

# Visualization libraries
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Data processing
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    # Map settings
    map_center: Tuple[float, float] = (40.0, -89.0)  # Center of MISO region
    map_zoom: int = 6
    
    # Color settings
    risk_colormap: str = "YlOrRd"
    confidence_colormap: str = "viridis"
    
    # Display settings
    tooltip_precision: int = 3
    legend_precision: int = 2
    
    # Export settings
    map_width: int = 1200
    map_height: int = 800
    chart_width: int = 800
    chart_height: int = 600


class FoliumMapGenerator:
    """
    Generates interactive Folium maps with choropleth visualization,
    layer controls, and interactive tooltips.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize Folium map generator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.maps = {}  # Store generated maps by horizon
        
        logging.info("Folium map generator initialized")
    
    def create_risk_heatmap(self, 
                           grid_data: gpd.GeoDataFrame,
                           risk_data: pd.DataFrame,
                           horizons: List[int] = [12, 24, 36, 48]) -> folium.Map:
        """
        Create interactive risk heatmap with layer controls for multiple horizons.
        
        Args:
            grid_data: GeoDataFrame with hexagonal grid cells
            risk_data: DataFrame with risk scores and components by horizon
            horizons: List of forecast horizons to include
            
        Returns:
            Folium map with interactive layers
        """
        try:
            # Create base map
            base_map = folium.Map(
                location=self.config.map_center,
                zoom_start=self.config.map_zoom,
                tiles='OpenStreetMap'
            )
            
            # Add alternative tile layers
            folium.TileLayer('CartoDB positron', name='Light Map').add_to(base_map)
            folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(base_map)
            
            # Merge grid with risk data for each horizon
            for horizon in horizons:
                horizon_data = risk_data[risk_data['horizon_h'] == horizon].copy()
                
                if len(horizon_data) == 0:
                    logging.warning(f"No data found for horizon {horizon}h")
                    continue
                
                # Merge with grid geometry
                map_data = grid_data.merge(horizon_data, on='cell_id', how='inner')
                
                if len(map_data) == 0:
                    logging.warning(f"No matching grid cells for horizon {horizon}h")
                    continue
                
                # Create choropleth layer
                self._add_choropleth_layer(base_map, map_data, horizon)
            
            # Add layer control
            folium.LayerControl(position='topright', collapsed=False).add_to(base_map)
            
            # Add legend
            self._add_risk_legend(base_map, risk_data)
            
            # Add scale bar
            plugins.MeasureControl(position='bottomleft').add_to(base_map)
            
            # Add fullscreen button
            plugins.Fullscreen(position='topleft').add_to(base_map)
            
            logging.info(f"Created risk heatmap with {len(horizons)} horizon layers")
            
            return base_map
            
        except Exception as e:
            logging.error(f"Error creating risk heatmap: {e}")
            raise
    
    def _add_choropleth_layer(self, 
                             base_map: folium.Map, 
                             map_data: gpd.GeoDataFrame, 
                             horizon: int) -> None:
        """Add choropleth layer for specific forecast horizon"""
        
        # Calculate risk score statistics for color scaling
        risk_scores = map_data['final_risk'].dropna()
        if len(risk_scores) == 0:
            logging.warning(f"No valid risk scores for horizon {horizon}h")
            return
        
        vmin = risk_scores.quantile(0.05)  # Use 5th percentile to handle outliers
        vmax = risk_scores.quantile(0.95)  # Use 95th percentile to handle outliers
        
        # Create feature group for this horizon
        feature_group = folium.FeatureGroup(
            name=f'{horizon}h Forecast',
            show=horizon == 12  # Show 12h by default
        )
        
        # Add choropleth polygons
        for idx, row in map_data.iterrows():
            # Skip cells with missing data
            if pd.isna(row['final_risk']):
                continue
            
            # Normalize risk score for color mapping
            normalized_risk = (row['final_risk'] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            normalized_risk = np.clip(normalized_risk, 0, 1)
            
            # Get color from colormap
            color = self._get_color_from_risk(normalized_risk)
            
            # Create tooltip content
            tooltip_html = self._create_tooltip_html(row, horizon)
            
            # Add polygon to map
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                    'opacity': 0.8
                },
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(tooltip_html, max_width=400)
            ).add_to(feature_group)
        
        # Add feature group to map
        feature_group.add_to(base_map)
        
        logging.info(f"Added choropleth layer for {horizon}h forecast ({len(map_data)} cells)")
    
    def _get_color_from_risk(self, normalized_risk: float) -> str:
        """Get hex color from normalized risk score using YlOrRd colormap"""
        # YlOrRd colormap approximation
        if normalized_risk <= 0.2:
            # Light yellow
            return '#ffffcc'
        elif normalized_risk <= 0.4:
            # Yellow-orange
            return '#fed976'
        elif normalized_risk <= 0.6:
            # Orange
            return '#fd8d3c'
        elif normalized_risk <= 0.8:
            # Red-orange
            return '#e31a1c'
        else:
            # Dark red
            return '#800026'
    
    def _create_tooltip_html(self, row: pd.Series, horizon: int) -> str:
        """Create HTML content for interactive tooltips"""
        
        # Get top contributors (assuming they're stored in the data)
        top_contributors = self._get_top_contributors(row)
        
        # Format confidence
        confidence = row.get('confidence', 0.0)
        confidence_pct = confidence * 100
        
        # Create HTML content
        html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 300px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">
                Cell {row['cell_id']} - {horizon}h Forecast
            </h4>
            
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f0f0f0;">
                    <td style="padding: 4px; font-weight: bold;">Risk Score:</td>
                    <td style="padding: 4px; text-align: right;">
                        {row['final_risk']:.{self.config.tooltip_precision}f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Confidence:</td>
                    <td style="padding: 4px; text-align: right;">
                        {confidence_pct:.1f}%
                    </td>
                </tr>
            </table>
            
            <h5 style="margin: 10px 0 5px 0; color: #666;">Risk Components:</h5>
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                <tr>
                    <td style="padding: 2px;">Hazard:</td>
                    <td style="padding: 2px; text-align: right;">
                        {row.get('hazard_score', 0):.{self.config.tooltip_precision}f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 2px;">Exposure:</td>
                    <td style="padding: 2px; text-align: right;">
                        {row.get('exposure_score', 0):.{self.config.tooltip_precision}f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 2px;">Vulnerability:</td>
                    <td style="padding: 2px; text-align: right;">
                        {row.get('vulnerability_score', 0):.{self.config.tooltip_precision}f}
                    </td>
                </tr>
            </table>
            
            <h5 style="margin: 10px 0 5px 0; color: #666;">Top Contributors:</h5>
            <ul style="margin: 0; padding-left: 15px; font-size: 11px;">
        """
        
        for contributor in top_contributors[:3]:
            html += f"<li>{contributor}</li>"
        
        html += """
            </ul>
            
            <h5 style="margin: 10px 0 5px 0; color: #666;">Weather Inputs:</h5>
            <table style="width: 100%; border-collapse: collapse; font-size: 10px;">
        """
        
        # Add weather data if available
        weather_fields = [
            ('temp_2m', 'Temperature', '°F'),
            ('heat_index', 'Heat Index', '°F'),
            ('wind_speed', 'Wind Speed', 'mph'),
            ('wind_gust', 'Wind Gust', 'mph'),
            ('precip_rate', 'Precipitation', 'mm/h')
        ]
        
        for field, label, unit in weather_fields:
            if field in row and not pd.isna(row[field]):
                html += f"""
                <tr>
                    <td style="padding: 1px;">{label}:</td>
                    <td style="padding: 1px; text-align: right;">
                        {row[field]:.1f} {unit}
                    </td>
                </tr>
                """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _get_top_contributors(self, row: pd.Series) -> List[str]:
        """Get top contributing factors for a grid cell"""
        
        # If top_contributors is already in the data, use it
        if 'top_contributors' in row and isinstance(row['top_contributors'], list):
            return row['top_contributors']
        
        # Otherwise, calculate from component scores
        contributors = []
        
        # Check hazard components
        hazard_components = {
            'Thermal Stress': row.get('thermal_stress', 0),
            'Wind Stress': row.get('wind_stress', 0),
            'Precipitation Stress': row.get('precip_stress', 0),
            'Storm Conditions': row.get('storm_proxy', 0)
        }
        
        # Check exposure components
        exposure_components = {
            'Population Density': row.get('normalized_pop_density', 0),
            'Load Factor': row.get('load_factor', 0)
        }
        
        # Check vulnerability components
        vulnerability_components = {
            'Renewable Share': row.get('renewable_share', 0),
            'Transmission Scarcity': row.get('transmission_scarcity', 0)
        }
        
        # Combine all components
        all_components = {**hazard_components, **exposure_components, **vulnerability_components}
        
        # Sort by value and return top contributors
        sorted_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)
        contributors = [name for name, value in sorted_components if value > 0]
        
        return contributors[:3] if contributors else ['Data not available']
    
    def _add_risk_legend(self, base_map: folium.Map, risk_data: pd.DataFrame) -> None:
        """Add color legend for risk scores"""
        
        # Calculate risk score range
        risk_scores = risk_data['final_risk'].dropna()
        if len(risk_scores) == 0:
            return
        
        vmin = risk_scores.quantile(0.05)
        vmax = risk_scores.quantile(0.95)
        
        # Create legend HTML
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4 style="margin: 0 0 10px 0;">Risk Score</h4>
        <div style="display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <div style="width: 20px; height: 15px; background-color: #800026; margin-right: 5px;"></div>
                <span>High ({vmax:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <div style="width: 20px; height: 15px; background-color: #fd8d3c; margin-right: 5px;"></div>
                <span>Medium</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <div style="width: 20px; height: 15px; background-color: #ffffcc; margin-right: 5px;"></div>
                <span>Low ({vmin:.2f})</span>
            </div>
        </div>
        </div>
        """
        
        base_map.get_root().html.add_child(folium.Element(legend_html))
    
    def export_map(self, 
                   map_obj: folium.Map, 
                   output_path: str,
                   title: str = "MISO Weather-Stress Heatmap") -> None:
        """
        Export Folium map to HTML file.
        
        Args:
            map_obj: Folium map object
            output_path: Path for output HTML file
            title: Map title for HTML page
        """
        try:
            # Add title to map
            title_html = f"""
            <h2 style="position: absolute; top: 10px; left: 50px; z-index: 9999; 
                       background-color: rgba(255,255,255,0.8); padding: 10px; 
                       border-radius: 5px; margin: 0;">
                {title}
            </h2>
            """
            map_obj.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            map_obj.save(output_path)
            
            logging.info(f"Map exported to: {output_path}")
            
        except Exception as e:
            logging.error(f"Error exporting map: {e}")
            raise


class SummaryVisualizationGenerator:
    """
    Generates summary visualizations including tables, charts, and ablation analysis.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize summary visualization generator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        logging.info("Summary visualization generator initialized")
    
    def create_top_risk_table(self, 
                             risk_data: pd.DataFrame,
                             horizon: int,
                             n_top: int = 10) -> pd.DataFrame:
        """
        Create table of top-N highest risk cells with contributing factors.
        
        Args:
            risk_data: DataFrame with risk scores and components
            horizon: Forecast horizon to analyze
            n_top: Number of top cells to include
            
        Returns:
            DataFrame with top risk cells and details
        """
        try:
            # Filter data for specific horizon
            horizon_data = risk_data[risk_data['horizon_h'] == horizon].copy()
            
            if len(horizon_data) == 0:
                logging.warning(f"No data found for horizon {horizon}h")
                return pd.DataFrame()
            
            # Sort by risk score and get top N
            top_cells = horizon_data.nlargest(n_top, 'final_risk')
            
            # Create summary table
            summary_table = pd.DataFrame({
                'Rank': range(1, len(top_cells) + 1),
                'Cell ID': top_cells['cell_id'],
                'Risk Score': top_cells['final_risk'].round(3),
                'Hazard': top_cells.get('hazard_score', 0).round(3),
                'Exposure': top_cells.get('exposure_score', 0).round(3),
                'Vulnerability': top_cells.get('vulnerability_score', 0).round(3),
                'Confidence': (top_cells.get('confidence', 0) * 100).round(1),
                'Top Contributor': top_cells.apply(self._get_primary_contributor, axis=1)
            })
            
            logging.info(f"Created top-{n_top} risk table for {horizon}h forecast")
            
            return summary_table
            
        except Exception as e:
            logging.error(f"Error creating top risk table: {e}")
            raise
    
    def _get_primary_contributor(self, row: pd.Series) -> str:
        """Get primary contributing factor for a cell"""
        
        # Check which component has highest score
        components = {
            'Hazard': row.get('hazard_score', 0),
            'Exposure': row.get('exposure_score', 0),
            'Vulnerability': row.get('vulnerability_score', 0)
        }
        
        primary = max(components.items(), key=lambda x: x[1])
        return primary[0]
    
    def create_risk_component_chart(self, 
                                   risk_data: pd.DataFrame,
                                   cell_ids: List[str],
                                   horizon: int) -> go.Figure:
        """
        Create Plotly bar chart showing risk component breakdown for selected cells.
        
        Args:
            risk_data: DataFrame with risk scores and components
            cell_ids: List of cell IDs to include in chart
            horizon: Forecast horizon to analyze
            
        Returns:
            Plotly figure object
        """
        try:
            # Filter data
            horizon_data = risk_data[
                (risk_data['horizon_h'] == horizon) & 
                (risk_data['cell_id'].isin(cell_ids))
            ].copy()
            
            if len(horizon_data) == 0:
                logging.warning(f"No data found for selected cells at {horizon}h")
                return go.Figure()
            
            # Prepare data for stacked bar chart
            fig = go.Figure()
            
            # Add bars for each component
            fig.add_trace(go.Bar(
                name='Hazard',
                x=horizon_data['cell_id'],
                y=horizon_data.get('hazard_score', 0),
                marker_color='#ff7f0e'
            ))
            
            fig.add_trace(go.Bar(
                name='Exposure',
                x=horizon_data['cell_id'],
                y=horizon_data.get('exposure_score', 0),
                marker_color='#2ca02c'
            ))
            
            fig.add_trace(go.Bar(
                name='Vulnerability',
                x=horizon_data['cell_id'],
                y=horizon_data.get('vulnerability_score', 0),
                marker_color='#d62728'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Risk Component Breakdown - {horizon}h Forecast',
                xaxis_title='Grid Cell ID',
                yaxis_title='Component Score',
                barmode='group',
                width=self.config.chart_width,
                height=self.config.chart_height,
                showlegend=True
            )
            
            logging.info(f"Created risk component chart for {len(cell_ids)} cells")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating risk component chart: {e}")
            raise
    
    def create_risk_evolution_chart(self, 
                                   risk_data: pd.DataFrame,
                                   cell_ids: List[str],
                                   horizons: List[int] = [12, 24, 36, 48]) -> go.Figure:
        """
        Create line chart showing risk evolution across forecast horizons.
        
        Args:
            risk_data: DataFrame with risk scores across horizons
            cell_ids: List of cell IDs to include
            horizons: List of forecast horizons
            
        Returns:
            Plotly figure object
        """
        try:
            # Filter data for selected cells
            selected_data = risk_data[risk_data['cell_id'].isin(cell_ids)].copy()
            
            if len(selected_data) == 0:
                logging.warning("No data found for selected cells")
                return go.Figure()
            
            # Create line chart
            fig = go.Figure()
            
            # Add line for each cell
            for cell_id in cell_ids:
                cell_data = selected_data[selected_data['cell_id'] == cell_id]
                
                if len(cell_data) == 0:
                    continue
                
                # Sort by horizon
                cell_data = cell_data.sort_values('horizon_h')
                
                fig.add_trace(go.Scatter(
                    x=cell_data['horizon_h'],
                    y=cell_data['final_risk'],
                    mode='lines+markers',
                    name=f'Cell {cell_id}',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            # Update layout
            fig.update_layout(
                title='Risk Score Evolution Across Forecast Horizons',
                xaxis_title='Forecast Horizon (hours)',
                yaxis_title='Risk Score',
                width=self.config.chart_width,
                height=self.config.chart_height,
                showlegend=True,
                xaxis=dict(tickmode='array', tickvals=horizons)
            )
            
            logging.info(f"Created risk evolution chart for {len(cell_ids)} cells")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating risk evolution chart: {e}")
            raise
    
    def create_ablation_analysis_chart(self, 
                                      risk_data: pd.DataFrame,
                                      ablation_results: Dict[str, pd.DataFrame],
                                      horizon: int) -> go.Figure:
        """
        Create chart showing risk sensitivity to component removal (ablation analysis).
        
        Args:
            risk_data: Original risk data
            ablation_results: Dictionary with ablation results for each component
            horizon: Forecast horizon to analyze
            
        Returns:
            Plotly figure object
        """
        try:
            # Filter original data for horizon
            original_data = risk_data[risk_data['horizon_h'] == horizon].copy()
            
            if len(original_data) == 0:
                logging.warning(f"No original data for horizon {horizon}h")
                return go.Figure()
            
            # Calculate risk changes for each component removal
            component_impacts = {}
            
            for component, ablation_data in ablation_results.items():
                if len(ablation_data) == 0:
                    continue
                
                # Merge with original data
                merged = original_data.merge(
                    ablation_data[['cell_id', 'final_risk']], 
                    on='cell_id', 
                    suffixes=('_original', '_ablated')
                )
                
                # Calculate impact (original - ablated)
                impact = merged['final_risk_original'] - merged['final_risk_ablated']
                component_impacts[component] = impact.mean()
            
            if not component_impacts:
                logging.warning("No ablation results to display")
                return go.Figure()
            
            # Create bar chart
            components = list(component_impacts.keys())
            impacts = list(component_impacts.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=components,
                    y=impacts,
                    marker_color=['#ff7f0e', '#2ca02c', '#d62728'][:len(components)]
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f'Component Importance (Ablation Analysis) - {horizon}h Forecast',
                xaxis_title='Risk Component',
                yaxis_title='Average Risk Reduction When Removed',
                width=self.config.chart_width,
                height=self.config.chart_height
            )
            
            logging.info(f"Created ablation analysis chart for {len(components)} components")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating ablation analysis chart: {e}")
            raise


class VisualizationSystem:
    """
    Main visualization system that coordinates map generation and summary visualizations.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization system.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize components
        self.map_generator = FoliumMapGenerator(self.config)
        self.summary_generator = SummaryVisualizationGenerator(self.config)
        
        # Storage for generated visualizations
        self.maps = {}
        self.charts = {}
        self.tables = {}
        
        logging.info("Visualization system initialized")
    
    def generate_all_visualizations(self, 
                                   grid_data: gpd.GeoDataFrame,
                                   risk_data: pd.DataFrame,
                                   output_dir: str = "output",
                                   horizons: List[int] = [12, 24, 36, 48]) -> Dict[str, Any]:
        """
        Generate complete set of visualizations for the risk assessment.
        
        Args:
            grid_data: GeoDataFrame with hexagonal grid
            risk_data: DataFrame with risk scores and components
            output_dir: Directory for output files
            horizons: List of forecast horizons
            
        Returns:
            Dictionary with paths to generated files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            results = {
                'maps': {},
                'charts': {},
                'tables': {},
                'summary': {}
            }
            
            # Generate interactive risk heatmap
            logging.info("Generating interactive risk heatmap...")
            risk_map = self.map_generator.create_risk_heatmap(
                grid_data, risk_data, horizons
            )
            
            # Export main map
            main_map_path = os.path.join(output_dir, "miso_risk_heatmap.html")
            self.map_generator.export_map(risk_map, main_map_path)
            results['maps']['main_heatmap'] = main_map_path
            
            # Generate summary visualizations for each horizon
            for horizon in horizons:
                logging.info(f"Generating summary visualizations for {horizon}h forecast...")
                
                # Top risk table
                top_table = self.summary_generator.create_top_risk_table(
                    risk_data, horizon, n_top=10
                )
                
                if len(top_table) > 0:
                    table_path = os.path.join(output_dir, f"top_risk_cells_{horizon}h.csv")
                    top_table.to_csv(table_path, index=False)
                    results['tables'][f'{horizon}h'] = table_path
                    
                    # Get top 5 cells for detailed charts
                    top_cell_ids = top_table['Cell ID'].head(5).tolist()
                    
                    # Risk component breakdown chart
                    component_chart = self.summary_generator.create_risk_component_chart(
                        risk_data, top_cell_ids, horizon
                    )
                    
                    component_chart_path = os.path.join(output_dir, f"risk_components_{horizon}h.html")
                    component_chart.write_html(component_chart_path)
                    results['charts'][f'components_{horizon}h'] = component_chart_path
            
            # Generate risk evolution chart for top cells across all horizons
            if len(horizons) > 1:
                # Get consistently high-risk cells across horizons
                all_top_cells = set()
                for horizon in horizons:
                    horizon_data = risk_data[risk_data['horizon_h'] == horizon]
                    if len(horizon_data) > 0:
                        top_5 = horizon_data.nlargest(5, 'final_risk')['cell_id'].tolist()
                        all_top_cells.update(top_5)
                
                if all_top_cells:
                    evolution_chart = self.summary_generator.create_risk_evolution_chart(
                        risk_data, list(all_top_cells)[:10], horizons  # Limit to 10 cells
                    )
                    
                    evolution_chart_path = os.path.join(output_dir, "risk_evolution.html")
                    evolution_chart.write_html(evolution_chart_path)
                    results['charts']['evolution'] = evolution_chart_path
            
            # Generate summary statistics
            summary_stats = self._generate_summary_statistics(risk_data, horizons)
            summary_path = os.path.join(output_dir, "summary_statistics.json")
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            results['summary']['statistics'] = summary_path
            
            logging.info("✓ All visualizations generated successfully")
            
            return results
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            raise
    
    def _generate_summary_statistics(self, 
                                   risk_data: pd.DataFrame,
                                   horizons: List[int]) -> Dict[str, Any]:
        """Generate summary statistics for risk assessment"""
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_cells': len(risk_data['cell_id'].unique()),
            'horizons_analyzed': horizons,
            'risk_statistics': {}
        }
        
        for horizon in horizons:
            horizon_data = risk_data[risk_data['horizon_h'] == horizon]
            
            if len(horizon_data) > 0:
                risk_scores = horizon_data['final_risk'].dropna()
                
                summary['risk_statistics'][f'{horizon}h'] = {
                    'count': len(risk_scores),
                    'mean': float(risk_scores.mean()),
                    'std': float(risk_scores.std()),
                    'min': float(risk_scores.min()),
                    'max': float(risk_scores.max()),
                    'median': float(risk_scores.median()),
                    'q95': float(risk_scores.quantile(0.95)),
                    'high_risk_cells': len(risk_scores[risk_scores > risk_scores.quantile(0.9)])
                }
        
        return summary


class AblationAnalysisEngine:
    """
    Performs ablation analysis to show risk sensitivity to component removal.
    """
    
    def __init__(self, risk_scoring_engine):
        """
        Initialize ablation analysis engine.
        
        Args:
            risk_scoring_engine: Instance of RiskScoringEngine for recalculation
        """
        self.risk_engine = risk_scoring_engine
        
        logging.info("Ablation analysis engine initialized")
    
    def perform_ablation_analysis(self, 
                                 combined_data: pd.DataFrame,
                                 components_to_test: List[str] = ['hazard', 'exposure', 'vulnerability']) -> Dict[str, pd.DataFrame]:
        """
        Perform ablation analysis by removing individual components and recalculating risk.
        
        Args:
            combined_data: DataFrame with all risk components and scores
            components_to_test: List of components to test removal
            
        Returns:
            Dictionary with ablation results for each component
        """
        try:
            ablation_results = {}
            
            for component in components_to_test:
                logging.info(f"Performing ablation analysis for {component} component...")
                
                # Create modified data with component removed (set to 0)
                modified_data = combined_data.copy()
                
                if component == 'hazard':
                    # Remove hazard component (set hazard score to 0)
                    modified_data['hazard_score'] = 0.0
                elif component == 'exposure':
                    # Remove exposure component (set exposure score to 0)
                    modified_data['exposure_score'] = 0.0
                elif component == 'vulnerability':
                    # Remove vulnerability component (set vulnerability score to 0)
                    modified_data['vulnerability_score'] = 0.0
                
                # Recalculate final risk scores without this component
                ablated_risk = self._recalculate_final_risk(modified_data)
                
                # Store results
                ablation_results[component] = ablated_risk
                
                logging.info(f"Completed ablation analysis for {component}")
            
            return ablation_results
            
        except Exception as e:
            logging.error(f"Error in ablation analysis: {e}")
            raise
    
    def _recalculate_final_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Recalculate final risk scores using modified component scores"""
        
        # Use the same formula as the risk engine: Risk = zscore(α×H + β×E + γ×V)
        weights = self.risk_engine.weights
        
        # Calculate weighted combination
        combined_score = (
            weights.alpha * data['hazard_score'] +
            weights.beta * data['exposure_score'] +
            weights.gamma * data['vulnerability_score']
        )
        
        # Apply z-score normalization within each horizon group
        result_data = data[['cell_id', 'horizon_h']].copy()
        
        for horizon in data['horizon_h'].unique():
            horizon_mask = data['horizon_h'] == horizon
            horizon_scores = combined_score[horizon_mask]
            
            if len(horizon_scores) > 1 and horizon_scores.std() > 0:
                # Calculate z-scores
                z_scores = (horizon_scores - horizon_scores.mean()) / horizon_scores.std()
            else:
                # Handle edge case where all scores are the same
                z_scores = pd.Series(0.0, index=horizon_scores.index)
            
            result_data.loc[horizon_mask, 'final_risk'] = z_scores
        
        return result_data
    
    def calculate_component_importance(self, 
                                     original_data: pd.DataFrame,
                                     ablation_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate importance of each component based on ablation results.
        
        Args:
            original_data: Original risk data
            ablation_results: Results from ablation analysis
            
        Returns:
            DataFrame with component importance metrics
        """
        try:
            importance_results = []
            
            for component, ablated_data in ablation_results.items():
                # Merge original and ablated data
                merged = original_data.merge(
                    ablated_data[['cell_id', 'horizon_h', 'final_risk']], 
                    on=['cell_id', 'horizon_h'], 
                    suffixes=('_original', '_ablated')
                )
                
                # Calculate impact metrics
                risk_difference = merged['final_risk_original'] - merged['final_risk_ablated']
                
                importance_metrics = {
                    'component': component,
                    'mean_impact': risk_difference.mean(),
                    'std_impact': risk_difference.std(),
                    'max_impact': risk_difference.max(),
                    'min_impact': risk_difference.min(),
                    'median_impact': risk_difference.median(),
                    'cells_affected': len(risk_difference[risk_difference.abs() > 0.01]),  # Cells with >0.01 change
                    'total_cells': len(risk_difference)
                }
                
                importance_results.append(importance_metrics)
            
            importance_df = pd.DataFrame(importance_results)
            
            # Sort by mean impact (descending)
            importance_df = importance_df.sort_values('mean_impact', ascending=False)
            
            logging.info("Component importance analysis completed")
            
            return importance_df
            
        except Exception as e:
            logging.error(f"Error calculating component importance: {e}")
            raise
    
    def create_ablation_summary_table(self, 
                                     importance_df: pd.DataFrame) -> pd.DataFrame:
        """Create formatted summary table for ablation analysis results"""
        
        summary_table = pd.DataFrame({
            'Component': importance_df['component'].str.title(),
            'Mean Impact': importance_df['mean_impact'].round(4),
            'Max Impact': importance_df['max_impact'].round(4),
            'Cells Affected': importance_df['cells_affected'],
            'Affected %': (importance_df['cells_affected'] / importance_df['total_cells'] * 100).round(1)
        })
        
        return summary_table


# Add ablation analysis to the main VisualizationSystem class
def add_ablation_to_visualization_system():
    """Extend VisualizationSystem with ablation analysis capabilities"""
    
    def generate_ablation_analysis(self, 
                                  combined_data: pd.DataFrame,
                                  risk_scoring_engine,
                                  output_dir: str = "output",
                                  horizons: List[int] = [12, 24, 36, 48]) -> Dict[str, str]:
        """
        Generate ablation analysis visualizations.
        
        Args:
            combined_data: DataFrame with all risk components
            risk_scoring_engine: RiskScoringEngine instance
            output_dir: Output directory
            horizons: Forecast horizons to analyze
            
        Returns:
            Dictionary with paths to generated ablation files
        """
        try:
            # Initialize ablation engine
            ablation_engine = AblationAnalysisEngine(risk_scoring_engine)
            
            results = {}
            
            # Perform ablation analysis for each horizon
            for horizon in horizons:
                logging.info(f"Generating ablation analysis for {horizon}h forecast...")
                
                # Filter data for this horizon
                horizon_data = combined_data[combined_data['horizon_h'] == horizon].copy()
                
                if len(horizon_data) == 0:
                    logging.warning(f"No data for horizon {horizon}h")
                    continue
                
                # Perform ablation analysis
                ablation_results = ablation_engine.perform_ablation_analysis(horizon_data)
                
                # Calculate component importance
                importance_df = ablation_engine.calculate_component_importance(
                    horizon_data, ablation_results
                )
                
                # Create ablation chart
                ablation_chart = self.summary_generator.create_ablation_analysis_chart(
                    horizon_data, ablation_results, horizon
                )
                
                # Save chart
                chart_path = os.path.join(output_dir, f"ablation_analysis_{horizon}h.html")
                ablation_chart.write_html(chart_path)
                results[f'ablation_chart_{horizon}h'] = chart_path
                
                # Create and save summary table
                summary_table = ablation_engine.create_ablation_summary_table(importance_df)
                table_path = os.path.join(output_dir, f"ablation_summary_{horizon}h.csv")
                summary_table.to_csv(table_path, index=False)
                results[f'ablation_table_{horizon}h'] = table_path
            
            logging.info("✓ Ablation analysis completed")
            
            return results
            
        except Exception as e:
            logging.error(f"Error generating ablation analysis: {e}")
            raise
    
    # Add method to VisualizationSystem class
    VisualizationSystem.generate_ablation_analysis = generate_ablation_analysis

# Apply the extension
add_ablation_to_visualization_system()


# Validation and Testing Functions
class VisualizationValidation:
    """Validation functions for visualization system"""
    
    @staticmethod
    def validate_map_generation(map_obj: folium.Map, 
                               expected_layers: int) -> Dict[str, bool]:
        """Validate Folium map generation"""
        
        validation_results = {}
        
        try:
            # Check map object type
            validation_results['is_folium_map'] = isinstance(map_obj, folium.Map)
            
            # Check for layer control
            has_layer_control = any(
                isinstance(child, folium.LayerControl) 
                for child in map_obj._children.values()
            )
            validation_results['has_layer_control'] = has_layer_control
            
            # Check for feature groups (layers)
            feature_groups = [
                child for child in map_obj._children.values() 
                if isinstance(child, folium.FeatureGroup)
            ]
            validation_results['correct_layer_count'] = len(feature_groups) == expected_layers
            
            # Check for plugins (scale, fullscreen)
            has_plugins = any(
                'MeasureControl' in str(type(child)) or 'Fullscreen' in str(type(child))
                for child in map_obj._children.values()
            )
            validation_results['has_plugins'] = has_plugins
            
        except Exception as e:
            logging.error(f"Map validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    @staticmethod
    def validate_chart_generation(fig: go.Figure) -> Dict[str, bool]:
        """Validate Plotly chart generation"""
        
        validation_results = {}
        
        try:
            # Check figure type
            validation_results['is_plotly_figure'] = isinstance(fig, go.Figure)
            
            # Check for data traces
            validation_results['has_data'] = len(fig.data) > 0
            
            # Check for title
            validation_results['has_title'] = bool(fig.layout.title.text if fig.layout.title else False)
            
            # Check for axis labels
            validation_results['has_axis_labels'] = bool(
                fig.layout.xaxis.title.text and fig.layout.yaxis.title.text
            )
            
        except Exception as e:
            logging.error(f"Chart validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    @staticmethod
    def validate_table_generation(df: pd.DataFrame, 
                                 expected_columns: List[str]) -> Dict[str, bool]:
        """Validate table generation"""
        
        validation_results = {}
        
        try:
            # Check DataFrame type
            validation_results['is_dataframe'] = isinstance(df, pd.DataFrame)
            
            # Check for data
            validation_results['has_data'] = len(df) > 0
            
            # Check for expected columns
            missing_columns = set(expected_columns) - set(df.columns)
            validation_results['has_required_columns'] = len(missing_columns) == 0
            
            # Check for reasonable data ranges
            if 'Risk Score' in df.columns:
                risk_scores = df['Risk Score'].dropna()
                validation_results['reasonable_risk_range'] = (
                    len(risk_scores) > 0 and 
                    risk_scores.min() >= -5 and 
                    risk_scores.max() <= 5
                )
            
        except Exception as e:
            logging.error(f"Table validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results


# Test functions for the visualization system
class VisualizationTests:
    """Test suite for visualization system"""
    
    def __init__(self, viz_system: VisualizationSystem):
        self.viz_system = viz_system
        self.test_results = {}
    
    def run_all_tests(self, 
                     grid_data: gpd.GeoDataFrame,
                     risk_data: pd.DataFrame) -> Dict[str, bool]:
        """Run all visualization tests"""
        
        logging.info("Running visualization system tests...")
        
        tests = [
            ('test_map_generation', lambda: self.test_map_generation(grid_data, risk_data)),
            ('test_table_generation', lambda: self.test_table_generation(risk_data)),
            ('test_chart_generation', lambda: self.test_chart_generation(risk_data)),
            ('test_export_functionality', lambda: self.test_export_functionality()),
            ('test_tooltip_content', lambda: self.test_tooltip_content(risk_data)),
            ('test_color_scaling', lambda: self.test_color_scaling(risk_data))
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✓ PASS" if result else "✗ FAIL"
                logging.info(f"{status}: {test_name}")
            except Exception as e:
                self.test_results[test_name] = False
                logging.error(f"✗ ERROR: {test_name} - {e}")
        
        # Summary
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        logging.info(f"Visualization Test Results: {passed}/{total} passed")
        
        return self.test_results
    
    def test_map_generation(self, 
                           grid_data: gpd.GeoDataFrame, 
                           risk_data: pd.DataFrame) -> bool:
        """Test interactive map generation"""
        
        try:
            # Generate map
            test_map = self.viz_system.map_generator.create_risk_heatmap(
                grid_data, risk_data, [12, 24]
            )
            
            # Validate map
            validation = VisualizationValidation.validate_map_generation(test_map, 2)
            
            return all(validation.values())
            
        except Exception:
            return False
    
    def test_table_generation(self, risk_data: pd.DataFrame) -> bool:
        """Test summary table generation"""
        
        try:
            # Generate table
            test_table = self.viz_system.summary_generator.create_top_risk_table(
                risk_data, 12, n_top=5
            )
            
            # Validate table
            expected_columns = ['Rank', 'Cell ID', 'Risk Score', 'Hazard', 'Exposure', 'Vulnerability']
            validation = VisualizationValidation.validate_table_generation(
                test_table, expected_columns
            )
            
            return all(validation.values())
            
        except Exception:
            return False
    
    def test_chart_generation(self, risk_data: pd.DataFrame) -> bool:
        """Test chart generation"""
        
        try:
            # Get sample cell IDs
            sample_cells = risk_data['cell_id'].unique()[:3]
            
            # Generate component chart
            component_chart = self.viz_system.summary_generator.create_risk_component_chart(
                risk_data, sample_cells, 12
            )
            
            # Validate chart
            validation = VisualizationValidation.validate_chart_generation(component_chart)
            
            return all(validation.values())
            
        except Exception:
            return False
    
    def test_export_functionality(self) -> bool:
        """Test file export functionality"""
        
        try:
            # Create simple test map
            test_map = folium.Map(location=[40, -89], zoom_start=6)
            
            # Test export
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                self.viz_system.map_generator.export_map(test_map, tmp.name)
                
                # Check file exists and has content
                return os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0
            
        except Exception:
            return False
    
    def test_tooltip_content(self, risk_data: pd.DataFrame) -> bool:
        """Test tooltip content generation"""
        
        try:
            # Get sample row
            if len(risk_data) == 0:
                return False
            
            sample_row = risk_data.iloc[0]
            
            # Generate tooltip
            tooltip_html = self.viz_system.map_generator._create_tooltip_html(sample_row, 12)
            
            # Check tooltip contains expected elements
            required_elements = ['Risk Score', 'Confidence', 'Risk Components', 'Top Contributors']
            
            return all(element in tooltip_html for element in required_elements)
            
        except Exception:
            return False
    
    def test_color_scaling(self, risk_data: pd.DataFrame) -> bool:
        """Test color scaling functionality"""
        
        try:
            # Test color generation for different risk levels
            test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            
            colors = [
                self.viz_system.map_generator._get_color_from_risk(val) 
                for val in test_values
            ]
            
            # Check all colors are valid hex codes
            valid_colors = all(
                isinstance(color, str) and color.startswith('#') and len(color) == 7
                for color in colors
            )
            
            # Check colors are different (progression)
            unique_colors = len(set(colors)) > 1
            
            return valid_colors and unique_colors
            
        except Exception:
            return False