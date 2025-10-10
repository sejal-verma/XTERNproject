#!/usr/bin/env python3
"""
Create Full MISO Weather-Stress Heatmap
Shows complete risk coverage across MISO territory
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import folium
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_full_miso_heatmap(horizon=24):
    """Create a complete heatmap showing all risk data"""
    
    print(f"🗺️  Creating Full MISO Weather-Stress Heatmap ({horizon}h forecast)...")
    
    # Load corrected spatial framework
    print("📍 Loading MISO spatial framework...")
    from core.spatial_framework import SpatialProcessingEngine
    
    config = {
        'runtime': {
            'crs': 'EPSG:4326',
            'hex_size_km': 40
        }
    }
    
    spatial_engine = SpatialProcessingEngine(config)
    footprint, grid = spatial_engine.initialize_spatial_framework()
    
    print(f"✅ Generated {len(grid)} hexagonal grid cells")
    
    # Load risk data
    print(f"📊 Loading risk data for {horizon}h horizon...")
    risk_data = pd.read_csv(f"output/risk_scores_{horizon}h.csv")
    
    # Merge grid geometry with risk data
    # Use grid coordinates instead of risk data coordinates to avoid missing data
    merged_data = grid.merge(risk_data[['cell_id', 'final_risk', 'hazard_score', 
                                      'exposure_score', 'vulnerability_score', 
                                      'temp_2m', 'wind_gust', 'precip_rate', 'confidence']], 
                           on='cell_id', how='left')
    
    # Fill missing risk scores with 0 (areas with no data)
    merged_data['final_risk'] = merged_data['final_risk'].fillna(0)
    merged_data['hazard_score'] = merged_data['hazard_score'].fillna(0)
    merged_data['exposure_score'] = merged_data['exposure_score'].fillna(0)
    merged_data['vulnerability_score'] = merged_data['vulnerability_score'].fillna(0)
    
    # Only show cells with actual risk data (non-zero risk)
    risk_cells = merged_data[merged_data['final_risk'] > 0]
    
    print(f"✅ Found {len(risk_cells)} cells with risk data")
    print(f"📊 Risk statistics:")
    print(f"   • Mean risk: {risk_cells['final_risk'].mean():.3f}")
    print(f"   • Max risk: {risk_cells['final_risk'].max():.3f}")
    print(f"   • High risk cells (>0.7): {(risk_cells['final_risk'] > 0.7).sum()}")
    
    # Create map centered on MISO territory
    center_lat = risk_cells.geometry.centroid.y.mean()
    center_lon = risk_cells.geometry.centroid.x.mean()
    
    print(f"📍 Map center: {center_lat:.2f}°N, {center_lon:.2f}°W")
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                background: rgba(255,255,255,0.95); padding: 12px; border: 2px solid #333; 
                border-radius: 8px; z-index: 9999; font-size: 20px; font-weight: bold; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        MISO Weather-Stress Heatmap - {horizon}h Forecast
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Enhanced color function for better heatmap visualization
    def get_risk_color(risk_score):
        """Get color based on risk score with smooth gradients"""
        if pd.isna(risk_score) or risk_score <= 0:
            return '#f0f0f0'  # Light gray for no data
        
        # Normalize risk score (0-3 range)
        normalized = min(max(risk_score, 0), 3) / 3
        
        if normalized <= 0.1:
            return '#00ff00'  # Bright green (very low)
        elif normalized <= 0.2:
            return '#80ff00'  # Yellow-green (low)
        elif normalized <= 0.3:
            return '#ffff00'  # Yellow (low-medium)
        elif normalized <= 0.4:
            return '#ffd700'  # Gold (medium)
        elif normalized <= 0.5:
            return '#ffa500'  # Orange (medium-high)
        elif normalized <= 0.6:
            return '#ff8c00'  # Dark orange (high)
        elif normalized <= 0.7:
            return '#ff6347'  # Tomato (high)
        elif normalized <= 0.8:
            return '#ff4500'  # Orange red (very high)
        elif normalized <= 0.9:
            return '#ff0000'  # Red (extreme)
        else:
            return '#8b0000'  # Dark red (maximum)
    
    # Add MISO boundary
    folium.GeoJson(
        footprint.iloc[0]['geometry'].__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': '#000080',
            'weight': 3,
            'fillOpacity': 0,
            'opacity': 1,
            'dashArray': '5, 5'
        },
        popup="MISO Territory Boundary"
    ).add_to(m)
    
    # Add all risk cells to create full heatmap coverage
    print(f"🎨 Adding {len(risk_cells)} risk cells to create heatmap...")
    
    for idx, row in risk_cells.iterrows():
        try:
            risk_score = row['final_risk']
            color = get_risk_color(risk_score)
            
            # Determine risk level for popup
            if risk_score >= 2.4:
                risk_level = "EXTREME"
            elif risk_score >= 1.2:
                risk_level = "HIGH"
            elif risk_score >= 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Create detailed popup
            popup_html = f"""
            <div style="font-family: Arial; font-size: 13px; min-width: 200px;">
                <div style="background: {color}; color: white; padding: 5px; margin: -5px -5px 5px -5px; 
                           font-weight: bold; text-align: center;">
                    {risk_level} RISK
                </div>
                <b>Cell:</b> {row['cell_id']}<br>
                <b>Risk Score:</b> {risk_score:.3f}<br>
                <hr style="margin: 5px 0;">
                <b>Risk Components:</b><br>
                • Hazard: {row.get('hazard_score', 0):.3f}<br>
                • Exposure: {row.get('exposure_score', 0):.3f}<br>
                • Vulnerability: {row.get('vulnerability_score', 0):.3f}<br>
                <hr style="margin: 5px 0;">
                <b>Weather Conditions:</b><br>
                • Temperature: {row.get('temp_2m', 'N/A'):.1f}°F<br>
                • Wind Gust: {row.get('wind_gust', 'N/A'):.1f} mph<br>
                • Precipitation: {row.get('precip_rate', 'N/A'):.1f} mm/h<br>
                • Confidence: {row.get('confidence', 'N/A'):.2f}
            </div>
            """
            
            # Add cell to map
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': '#333333',
                    'weight': 0.3,
                    'fillOpacity': 0.8,
                    'opacity': 0.6
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['cell_id']}: {risk_level} ({risk_score:.3f})"
            ).add_to(m)
            
        except Exception as e:
            print(f"Warning: Skipped cell {row.get('cell_id', 'unknown')}: {e}")
            continue
    
    # Add comprehensive legend
    legend_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; width: 200px; height: 280px; 
                background: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 8px;
                z-index: 9999; font-size: 12px; padding: 15px; font-family: Arial;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px; text-align: center;">
            Risk Level Legend
        </div>
        <div style="margin-bottom: 8px;">
            <span style="background: #00ff00; color: white; padding: 2px 6px; border-radius: 3px;">■</span> 
            Very Low (0.0-0.3)
        </div>
        <div style="margin-bottom: 8px;">
            <span style="background: #ffff00; color: black; padding: 2px 6px; border-radius: 3px;">■</span> 
            Low (0.3-0.6)
        </div>
        <div style="margin-bottom: 8px;">
            <span style="background: #ffa500; color: white; padding: 2px 6px; border-radius: 3px;">■</span> 
            Medium (0.6-1.2)
        </div>
        <div style="margin-bottom: 8px;">
            <span style="background: #ff4500; color: white; padding: 2px 6px; border-radius: 3px;">■</span> 
            High (1.2-2.4)
        </div>
        <div style="margin-bottom: 8px;">
            <span style="background: #8b0000; color: white; padding: 2px 6px; border-radius: 3px;">■</span> 
            Extreme (2.4+)
        </div>
        <hr style="margin: 10px 0;">
        <div style="font-size: 10px; color: #666;">
            • Blue dashed line = MISO boundary<br>
            • Click cells for detailed info<br>
            • {len(risk_cells)} cells with risk data
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add statistics box
    stats_html = f'''
    <div style="position: fixed; top: 80px; right: 20px; width: 220px; height: 140px; 
                background: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 8px;
                z-index: 9999; font-size: 12px; padding: 15px; font-family: Arial;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px; text-align: center;">
            Risk Statistics
        </div>
        <div><b>Forecast Horizon:</b> {horizon} hours</div>
        <div><b>Total Cells:</b> {len(risk_cells):,}</div>
        <div><b>Mean Risk:</b> {risk_cells['final_risk'].mean():.3f}</div>
        <div><b>Max Risk:</b> {risk_cells['final_risk'].max():.3f}</div>
        <div><b>High Risk Cells:</b> {(risk_cells['final_risk'] > 0.7).sum():,}</div>
        <div><b>Extreme Risk Cells:</b> {(risk_cells['final_risk'] > 2.4).sum():,}</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save full heatmap
    output_path = f"output/miso_full_heatmap_{horizon}h.html"
    m.save(output_path)
    
    print(f"\n✅ Full MISO heatmap saved: {output_path}")
    print(f"🎨 Heatmap shows {len(risk_cells)} cells with complete risk coverage")
    print(f"📊 Risk distribution:")
    print(f"   • Low risk (0-0.6): {(risk_cells['final_risk'] <= 0.6).sum()} cells")
    print(f"   • Medium risk (0.6-1.2): {((risk_cells['final_risk'] > 0.6) & (risk_cells['final_risk'] <= 1.2)).sum()} cells")
    print(f"   • High risk (1.2-2.4): {((risk_cells['final_risk'] > 1.2) & (risk_cells['final_risk'] <= 2.4)).sum()} cells")
    print(f"   • Extreme risk (>2.4): {(risk_cells['final_risk'] > 2.4).sum()} cells")
    
    return output_path

if __name__ == "__main__":
    try:
        # Create heatmaps for multiple horizons
        horizons = [12, 24, 36, 48]
        created_maps = []
        
        for horizon in horizons:
            print(f"\n{'='*60}")
            heatmap_path = create_full_miso_heatmap(horizon)
            created_maps.append(heatmap_path)
        
        print(f"\n🎉 Created {len(created_maps)} full heatmaps!")
        print(f"📁 Files created:")
        for map_path in created_maps:
            print(f"   • {os.path.basename(map_path)}")
        
        print(f"\n🌐 Open any file to see the complete weather-stress heatmap!")
        
    except Exception as e:
        print(f"❌ Error creating full heatmap: {e}")
        import traceback
        traceback.print_exc()