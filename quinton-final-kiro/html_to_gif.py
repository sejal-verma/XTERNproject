#!/usr/bin/env python3
"""
HTML to GIF Converter
Automatically converts HTML heatmaps to animated GIF using headless browser
"""

import os
import time
from pathlib import Path
from PIL import Image
import base64
import io

def html_to_gif_simple():
    """Simple approach using HTML to image conversion"""
    
    print("🎬 Converting HTML heatmaps to animated GIF...")
    
    try:
        # Try using playwright for HTML to image conversion
        try:
            from playwright.sync_api import sync_playwright
            return html_to_gif_playwright()
        except ImportError:
            print("📦 Playwright not available, trying selenium...")
            
        # Try using selenium
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            return html_to_gif_selenium()
        except ImportError:
            print("📦 Selenium not available, trying alternative method...")
            
        # Fallback to simple HTML parsing approach
        return html_to_gif_fallback()
        
    except Exception as e:
        print(f"❌ Error converting HTML to GIF: {e}")
        return None

def html_to_gif_playwright():
    """Convert HTML to GIF using Playwright (best option)"""
    
    from playwright.sync_api import sync_playwright
    
    print("🎭 Using Playwright for HTML to image conversion...")
    
    horizons = [12, 24, 36, 48]
    image_paths = []
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Set viewport size
        page.set_viewport_size({"width": 1200, "height": 800})
        
        for horizon in horizons:
            html_file = f"output/heatmaps/miso_full_heatmap_{horizon}h.html"
            
            if not os.path.exists(html_file):
                print(f"❌ HTML file not found: {html_file}")
                continue
            
            print(f"📸 Capturing screenshot for {horizon}h horizon...")
            
            try:
                # Load HTML file
                file_url = f"file://{os.path.abspath(html_file)}"
                page.goto(file_url)
                
                # Wait for map to load
                page.wait_for_timeout(3000)  # 3 seconds
                
                # Take screenshot
                screenshot_path = f"temp_screenshot_{horizon}h.png"
                page.screenshot(path=screenshot_path, full_page=False)
                
                image_paths.append(screenshot_path)
                print(f"   ✅ Screenshot saved: {screenshot_path}")
                
            except Exception as e:
                print(f"   ❌ Error capturing {horizon}h: {e}")
                continue
        
        browser.close()
    
    # Create GIF from screenshots
    if len(image_paths) >= 2:
        gif_path = create_gif_from_images(image_paths)
        
        # Clean up temporary files
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        return gif_path
    else:
        print("❌ Not enough screenshots captured")
        return None

def html_to_gif_selenium():
    """Convert HTML to GIF using Selenium (backup option)"""
    
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    print("🌐 Using Selenium for HTML to image conversion...")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1200,800")
    
    horizons = [12, 24, 36, 48]
    image_paths = []
    
    try:
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        
        for horizon in horizons:
            html_file = f"output/heatmaps/miso_full_heatmap_{horizon}h.html"
            
            if not os.path.exists(html_file):
                print(f"❌ HTML file not found: {html_file}")
                continue
            
            print(f"📸 Capturing screenshot for {horizon}h horizon...")
            
            try:
                # Load HTML file
                file_url = f"file://{os.path.abspath(html_file)}"
                driver.get(file_url)
                
                # Wait for map to load
                time.sleep(3)
                
                # Take screenshot
                screenshot_path = f"temp_screenshot_{horizon}h.png"
                driver.save_screenshot(screenshot_path)
                
                image_paths.append(screenshot_path)
                print(f"   ✅ Screenshot saved: {screenshot_path}")
                
            except Exception as e:
                print(f"   ❌ Error capturing {horizon}h: {e}")
                continue
        
        driver.quit()
        
    except Exception as e:
        print(f"❌ Selenium error: {e}")
        return None
    
    # Create GIF from screenshots
    if len(image_paths) >= 2:
        gif_path = create_gif_from_images(image_paths)
        
        # Clean up temporary files
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        return gif_path
    else:
        print("❌ Not enough screenshots captured")
        return None

def html_to_gif_fallback():
    """Fallback method using HTML parsing and matplotlib with state boundaries"""
    
    print("🔄 Using fallback method with matplotlib...")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
        import pandas as pd
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point, Polygon
        
        # Add src to path
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from core.spatial_framework import SpatialProcessingEngine
        
        # Initialize spatial framework
        config = {'runtime': {'crs': 'EPSG:4326', 'hex_size_km': 40}}
        spatial_engine = SpatialProcessingEngine(config)
        footprint, grid = spatial_engine.initialize_spatial_framework()
        
        # Create simplified state boundaries for MISO region
        state_boundaries = create_miso_state_boundaries()
        
        horizons = [12, 24, 36, 48]
        image_paths = []
        
        # Create color map
        colors = ['#2E8B57', '#9ACD32', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
        cmap = LinearSegmentedColormap.from_list('risk', colors, N=100)
        
        for horizon in horizons:
            print(f"🎨 Creating image for {horizon}h horizon...")
            
            try:
                # Load risk data
                risk_data = pd.read_csv(f"output/data/risk_scores_{horizon}h.csv")
                merged_data = grid.merge(risk_data[['cell_id', 'final_risk']], on='cell_id', how='left')
                merged_data['final_risk'] = merged_data['final_risk'].fillna(0)
                risk_cells = merged_data[merged_data['final_risk'] > 0]
                
                # Create figure with better layout
                fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                
                # Plot state boundaries first (background)
                for state_name, boundary in state_boundaries.items():
                    boundary_gdf = gpd.GeoDataFrame([1], geometry=[boundary], crs='EPSG:4326')
                    boundary_gdf.boundary.plot(ax=ax, color='gray', linewidth=1, alpha=0.6)
                
                # Plot MISO footprint boundary (thicker)
                footprint.boundary.plot(ax=ax, color='navy', linewidth=3, alpha=0.9, linestyle='--')
                
                # Plot risk cells
                if len(risk_cells) > 0:
                    risk_cells.plot(
                        ax=ax,
                        column='final_risk',
                        cmap=cmap,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.1,
                        vmin=0,
                        vmax=3
                    )
                
                # Add state labels
                state_centers = {
                    'Illinois': (40.0, -89.0),
                    'Indiana': (40.0, -86.0),
                    'Michigan': (44.5, -85.0),
                    'Wisconsin': (44.5, -90.0),
                    'Minnesota': (46.0, -94.0),
                    'Iowa': (42.0, -93.5),
                    'Missouri': (38.5, -92.5),
                    'Arkansas': (35.0, -92.0),
                    'N. Dakota': (47.5, -100.0)
                }
                
                for state, (lat, lon) in state_centers.items():
                    ax.text(lon, lat, state, fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           fontweight='bold', color='#333333')
                
                # Set extent with better margins
                bounds = footprint.total_bounds
                margin = 1.5
                ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
                
                # Remove axes but keep clean look
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Add enhanced title
                ax.set_title(f"MISO Weather-Stress Heatmap - {horizon}h Forecast", 
                           fontsize=18, fontweight='bold', pad=25)
                
                # Add subtitle with timestamp
                subtitle = f"Risk Assessment for MISO Territory • Generated {time.strftime('%Y-%m-%d %H:%M UTC')}"
                ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, ha='center',
                       fontsize=10, style='italic', color='#666666')
                
                # Add enhanced stats box
                if len(risk_cells) > 0:
                    high_risk = (risk_cells['final_risk'] > 0.7).sum()
                    extreme_risk = (risk_cells['final_risk'] > 2.4).sum()
                    
                    stats_text = f"""Risk Statistics:
Mean Risk: {risk_cells['final_risk'].mean():.2f}
Max Risk: {risk_cells['final_risk'].max():.2f}
High Risk Cells: {high_risk:,}
Extreme Risk: {extreme_risk:,}
Total Cells: {len(risk_cells):,}"""
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                
                # Add color bar legend
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=3))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
                cbar.set_label('Risk Score', fontsize=12, fontweight='bold')
                cbar.set_ticks([0, 0.6, 1.2, 2.4, 3.0])
                cbar.set_ticklabels(['Low\n(0.0)', 'Medium\n(0.6)', 'High\n(1.2)', 'Extreme\n(2.4)', '(3.0)'])
                
                # Add legend for boundaries
                legend_elements = [
                    plt.Line2D([0], [0], color='navy', linewidth=3, linestyle='--', label='MISO Territory'),
                    plt.Line2D([0], [0], color='gray', linewidth=1, label='State Boundaries')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                         bbox_to_anchor=(0.98, 0.02))
                
                # Save image with higher quality
                screenshot_path = f"temp_screenshot_{horizon}h.png"
                plt.savefig(screenshot_path, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.2)
                plt.close()
                
                image_paths.append(screenshot_path)
                print(f"   ✅ Enhanced image saved: {screenshot_path}")
                
            except Exception as e:
                print(f"   ❌ Error creating image for {horizon}h: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create GIF
        if len(image_paths) >= 2:
            gif_path = create_gif_from_images(image_paths)
            
            # Clean up
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            return gif_path
        else:
            return None
            
    except Exception as e:
        print(f"❌ Fallback method failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_miso_state_boundaries():
    """Create simplified state boundaries for MISO region"""
    
    from shapely.geometry import Polygon
    
    # Simplified state boundaries (approximate polygons for major MISO states)
    state_boundaries = {
        'Illinois': Polygon([
            (-91.5, 42.5), (-87.5, 42.5), (-87.5, 37.0), (-91.0, 37.0), (-91.5, 42.5)
        ]),
        'Indiana': Polygon([
            (-87.5, 41.8), (-84.8, 41.8), (-84.8, 37.8), (-87.5, 37.8), (-87.5, 41.8)
        ]),
        'Michigan': Polygon([
            (-90.4, 48.2), (-82.4, 48.2), (-82.4, 41.7), (-87.0, 41.7), (-90.4, 45.0), (-90.4, 48.2)
        ]),
        'Wisconsin': Polygon([
            (-92.9, 47.1), (-86.8, 47.1), (-86.8, 42.5), (-91.0, 42.5), (-92.9, 47.1)
        ]),
        'Minnesota': Polygon([
            (-97.2, 49.4), (-89.5, 49.4), (-89.5, 43.5), (-96.6, 43.5), (-97.2, 49.4)
        ]),
        'Iowa': Polygon([
            (-96.6, 43.5), (-90.1, 43.5), (-90.1, 40.4), (-96.6, 40.4), (-96.6, 43.5)
        ]),
        'Missouri': Polygon([
            (-95.8, 40.6), (-89.1, 40.6), (-89.1, 36.0), (-95.8, 36.0), (-95.8, 40.6)
        ]),
        'Arkansas': Polygon([
            (-94.6, 36.5), (-89.6, 36.5), (-89.6, 33.0), (-94.6, 33.0), (-94.6, 36.5)
        ]),
        'North Dakota': Polygon([
            (-104.0, 49.0), (-96.6, 49.0), (-96.6, 45.9), (-104.0, 45.9), (-104.0, 49.0)
        ])
    }
    
    return state_boundaries

def create_gif_from_images(image_paths):
    """Create animated GIF from image files"""
    
    print(f"🎞️  Creating animated GIF from {len(image_paths)} images...")
    
    try:
        # Load images
        images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
        
        if len(images) < 2:
            print("❌ Need at least 2 images to create GIF")
            return None
        
        # Create GIF
        gif_path = "output/miso_weather_stress_animation.gif"
        
        # Save as animated GIF
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1500,  # 1.5 seconds per frame
            loop=0,  # Loop forever
            optimize=True  # Optimize file size
        )
        
        # Get file info
        file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        
        print(f"✅ Animated GIF created: {gif_path}")
        print(f"   • File size: {file_size_mb:.1f} MB")
        print(f"   • Frames: {len(images)}")
        print(f"   • Duration: {1.5 * len(images)} seconds per loop")
        print(f"   • Frame rate: {len(images) / (1.5 * len(images)):.1f} fps")
        
        return gif_path
        
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        return None

def install_dependencies():
    """Install required dependencies for HTML to image conversion"""
    
    print("📦 Checking dependencies for HTML to GIF conversion...")
    
    # Check for playwright
    try:
        import playwright
        print("✅ Playwright is available")
        return "playwright"
    except ImportError:
        print("❌ Playwright not installed")
    
    # Check for selenium
    try:
        import selenium
        print("✅ Selenium is available")
        return "selenium"
    except ImportError:
        print("❌ Selenium not installed")
    
    print("\n💡 To install dependencies:")
    print("   For Playwright: pip install playwright && playwright install chromium")
    print("   For Selenium: pip install selenium (requires Chrome browser)")
    print("   Or use the fallback matplotlib method (no additional deps needed)")
    
    return "fallback"

def main():
    """Main function to convert HTML heatmaps to animated GIF"""
    
    print("🎬 HTML to GIF Converter for MISO Heatmaps")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if HTML files exist
    horizons = [12, 24, 36, 48]
    missing_files = []
    
    for horizon in horizons:
        html_file = f"output/heatmaps/miso_full_heatmap_{horizon}h.html"
        if not os.path.exists(html_file):
            missing_files.append(html_file)
    
    if missing_files:
        print(f"❌ Missing HTML files:")
        for file in missing_files:
            print(f"   • {file}")
        print("Run 'python create_full_heatmap.py' first to generate heatmaps")
        return
    
    # Check dependencies
    method = install_dependencies()
    
    # Convert HTML to GIF
    gif_path = html_to_gif_simple()
    
    if gif_path:
        print(f"\n🎉 Success! Animated GIF created: {gif_path}")
        print(f"🎯 Perfect for presentations - shows risk evolution over 12h → 24h → 36h → 48h")
        print(f"💡 Use this GIF in your slideshow to demonstrate temporal risk patterns")
    else:
        print(f"\n❌ Failed to create animated GIF")
        print(f"💡 Try the manual method in output/GIF_CREATION_INSTRUCTIONS.md")

if __name__ == "__main__":
    main()