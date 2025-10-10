# MISO Footprint and Hexagonal Grid Generation
# Implementation for Task 2: Spatial Framework

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon


class MISOFootprint:
    """Manages MISO territory boundary definition and validation"""
    
    # MISO states and territories (including partial states)
    MISO_STATES = {
        'full_states': [
            'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota', 
            'Iowa', 'North Dakota', 'Arkansas', 'Louisiana', 'Mississippi'
        ],
        'partial_states': {
            'Missouri': 'eastern',  # Eastern Missouri
            'Kentucky': 'western',  # Western Kentucky  
            'Ohio': 'northwestern', # Northwestern Ohio
            'Pennsylvania': 'northwestern', # Small northwestern portion
            'Texas': 'eastern_panhandle', # Eastern panhandle
            'Montana': 'eastern',   # Eastern Montana
            'South Dakota': 'southeastern' # Southeastern South Dakota
        },
        'canadian_territories': [
            'Manitoba'  # Southern Manitoba portion
        ]
    }
    
    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs
        self.footprint = None
        self.area_km2 = None
        
    def create_miso_footprint(self, use_simplified: bool = True) -> gpd.GeoDataFrame:
        """Create MISO footprint from state boundaries
        
        Args:
            use_simplified: If True, use simplified state boundaries for faster processing
            
        Returns:
            GeoDataFrame with MISO footprint polygon
        """
        try:
            # For demo mode, create a simplified MISO footprint
            footprint = self._create_demo_footprint()
            
            # Ensure correct CRS
            if footprint.crs != self.crs:
                footprint = footprint.to_crs(self.crs)
            
            # Calculate area
            footprint_utm = footprint.to_crs('EPSG:3857')  # Web Mercator for area calc
            self.area_km2 = footprint_utm.geometry.area.sum() / 1e6  # Convert m² to km²
            
            self.footprint = footprint
            
            logging.info(f"MISO footprint created: {self.area_km2:.0f} km²")
            logging.info(f"Footprint bounds: {footprint.total_bounds}")
            
            return footprint
            
        except Exception as e:
            logging.error(f"Error creating MISO footprint: {e}")
            raise
    
    def _create_demo_footprint(self) -> gpd.GeoDataFrame:
        """Create simplified demo footprint for development/testing"""
        # Approximate MISO footprint as a polygon covering the main region
        # Coordinates roughly covering IL, IN, MI, WI, MN, IA, MO, AR, LA, MS
        
        demo_coords = [
            (-98.0, 48.0),   # Northwest corner (ND)
            (-84.0, 48.0),   # Northeast corner (MI/Canada)
            (-82.0, 42.0),   # East side (OH border)
            (-85.0, 36.0),   # Southeast (KY/TN)
            (-92.0, 33.0),   # South (AR/LA)
            (-93.0, 29.5),   # Southwest (LA Gulf)
            (-98.0, 32.0),   # West side (TX panhandle)
            (-98.0, 48.0)    # Close polygon
        ]
        
        # Create polygon
        polygon = Polygon(demo_coords)
        
        # Create GeoDataFrame
        footprint = gpd.GeoDataFrame(
            {'region': ['MISO'], 'type': ['demo_footprint']},
            geometry=[polygon],
            crs=self.crs
        )
        
        logging.info("Created demo MISO footprint")
        return footprint
    
    def validate_footprint(self) -> bool:
        """Validate footprint geometry and properties"""
        if self.footprint is None:
            return False
            
        # Check geometry validity
        if not self.footprint.geometry.is_valid.all():
            logging.warning("Invalid geometries detected in footprint")
            return False
        
        # Check area is reasonable (MISO is ~2.4M km², allow wider range for demo)
        if not (1e6 < self.area_km2 < 8e6):
            logging.warning(f"Footprint area {self.area_km2:.0f} km² seems unreasonable")
            return False
        
        # Check CRS
        if self.footprint.crs != self.crs:
            logging.warning(f"Footprint CRS {self.footprint.crs} != expected {self.crs}")
            return False
        
        return True
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get footprint bounding box (minx, miny, maxx, maxy)"""
        if self.footprint is None:
            raise ValueError("Footprint not created yet")
        return tuple(self.footprint.total_bounds)
    
    def get_area_km2(self) -> float:
        """Get footprint area in km²"""
        return self.area_km2 if self.area_km2 else 0.0


class HexGridGenerator:
    """Generates uniform hexagonal analysis grid for spatial analysis"""
    
    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs
        self.grid = None
        self.hex_size_km = None
        
    def generate_hex_grid(self, footprint: gpd.GeoDataFrame, 
                         hex_size_km: float = 40) -> gpd.GeoDataFrame:
        """Generate hexagonal grid clipped to footprint
        
        Args:
            footprint: GeoDataFrame with boundary polygon
            hex_size_km: Approximate hex size in kilometers
            
        Returns:
            GeoDataFrame with hexagonal grid cells
        """
        try:
            self.hex_size_km = hex_size_km
            
            # Get footprint bounds
            minx, miny, maxx, maxy = footprint.total_bounds
            
            # Calculate hex size in degrees (approximate)
            # 1 degree latitude ≈ 111 km
            hex_size_deg = hex_size_km / 111.0
            
            # Generate hexagonal grid using a simple approach
            grid_cells = self._create_hex_grid_simple(minx, miny, maxx, maxy, hex_size_deg)
            
            # Create GeoDataFrame
            grid_gdf = gpd.GeoDataFrame(
                grid_cells,
                crs=self.crs
            )
            
            # Clip to footprint
            grid_clipped = gpd.clip(grid_gdf, footprint)
            
            # Add cell IDs and centroids
            grid_clipped = self._add_cell_metadata(grid_clipped)
            
            # Calculate actual hex areas
            grid_utm = grid_clipped.to_crs('EPSG:3857')
            grid_clipped['area_km2'] = grid_utm.geometry.area / 1e6
            
            self.grid = grid_clipped
            
            logging.info(f"Generated {len(grid_clipped)} hex cells")
            logging.info(f"Target hex size: {hex_size_km} km")
            logging.info(f"Actual hex area range: {grid_clipped['area_km2'].min():.1f} - {grid_clipped['area_km2'].max():.1f} km²")
            
            return grid_clipped
            
        except Exception as e:
            logging.error(f"Error generating hex grid: {e}")
            raise
    
    def _create_hex_grid_simple(self, minx: float, miny: float, 
                               maxx: float, maxy: float, 
                               hex_size_deg: float) -> List[Dict]:
        """Create simple hexagonal grid using regular spacing"""
        
        # Hexagon geometry parameters
        # For regular hexagon: width = 2 * radius, height = sqrt(3) * radius
        hex_width = hex_size_deg
        hex_height = hex_size_deg * np.sqrt(3) / 2
        
        # Row and column spacing
        row_spacing = hex_height * 0.75  # Overlap for tessellation
        col_spacing = hex_width
        
        grid_cells = []
        cell_id = 0
        
        # Generate grid
        y = miny
        row = 0
        
        while y < maxy:
            # Offset every other row for hexagonal tessellation
            x_offset = (col_spacing / 2) if row % 2 == 1 else 0
            x = minx + x_offset
            
            while x < maxx:
                # Create hexagon centered at (x, y)
                hex_polygon = self._create_hexagon(x, y, hex_size_deg / 2)
                
                grid_cells.append({
                    'cell_id': f"hex_{cell_id:04d}",
                    'row': row,
                    'col': int((x - minx - x_offset) / col_spacing),
                    'geometry': hex_polygon
                })
                
                cell_id += 1
                x += col_spacing
            
            y += row_spacing
            row += 1
        
        return grid_cells
    
    def _create_hexagon(self, center_x: float, center_y: float, 
                       radius: float) -> Polygon:
        """Create hexagon polygon centered at given point"""
        angles = np.linspace(0, 2 * np.pi, 7)  # 6 vertices + close
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)
        
        coords = list(zip(x_coords, y_coords))
        return Polygon(coords)
    
    def _add_cell_metadata(self, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add cell IDs, centroids, and other metadata"""
        # Ensure unique cell IDs
        if 'cell_id' not in grid_gdf.columns:
            grid_gdf['cell_id'] = [f"hex_{i:04d}" for i in range(len(grid_gdf))]
        
        # Calculate centroids
        centroids = grid_gdf.geometry.centroid
        grid_gdf['centroid_lon'] = centroids.x
        grid_gdf['centroid_lat'] = centroids.y
        
        # Add creation timestamp
        grid_gdf['created_at'] = datetime.now().isoformat()
        
        # Reset index to ensure clean DataFrame
        grid_gdf = grid_gdf.reset_index(drop=True)
        
        return grid_gdf
    
    def validate_grid(self) -> bool:
        """Validate grid geometry and properties"""
        if self.grid is None:
            return False
        
        # Check for valid geometries
        if not self.grid.geometry.is_valid.all():
            logging.warning("Invalid geometries detected in grid")
            return False
        
        # Check for duplicate cell IDs
        if self.grid['cell_id'].duplicated().any():
            logging.warning("Duplicate cell IDs detected")
            return False
        
        # Check reasonable number of cells (should be hundreds to low thousands)
        n_cells = len(self.grid)
        if not (50 < n_cells < 10000):
            logging.warning(f"Unexpected number of grid cells: {n_cells}")
            return False
        
        # Check area consistency
        area_cv = self.grid['area_km2'].std() / self.grid['area_km2'].mean()
        if area_cv > 0.5:  # Coefficient of variation > 50%
            logging.warning(f"High area variation in grid cells: CV = {area_cv:.2f}")
        
        return True
    
    def get_grid_summary(self) -> Dict:
        """Get summary statistics for the grid"""
        if self.grid is None:
            return {}
        
        return {
            'n_cells': len(self.grid),
            'target_hex_size_km': self.hex_size_km,
            'actual_area_km2': {
                'mean': self.grid['area_km2'].mean(),
                'std': self.grid['area_km2'].std(),
                'min': self.grid['area_km2'].min(),
                'max': self.grid['area_km2'].max()
            },
            'bounds': tuple(self.grid.total_bounds),
            'crs': str(self.grid.crs)
        }


class SpatialProcessingEngine:
    """Main interface for spatial processing operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.crs = config['runtime']['crs']
        self.hex_size_km = config['runtime']['hex_size_km']
        
        # Initialize components
        self.footprint_manager = MISOFootprint(crs=self.crs)
        self.grid_generator = HexGridGenerator(crs=self.crs)
        
        # Storage for results
        self.miso_footprint = None
        self.hex_grid = None
    
    def initialize_spatial_framework(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Initialize complete spatial framework: footprint + grid
        
        Returns:
            Tuple of (footprint_gdf, grid_gdf)
        """
        try:
            logging.info("Initializing spatial framework...")
            
            # Step 1: Create MISO footprint
            logging.info("Creating MISO footprint...")
            self.miso_footprint = self.footprint_manager.create_miso_footprint()
            
            # Validate footprint
            if not self.footprint_manager.validate_footprint():
                raise ValueError("Footprint validation failed")
            
            # Step 2: Generate hexagonal grid
            logging.info(f"Generating hexagonal grid ({self.hex_size_km} km spacing)...")
            self.hex_grid = self.grid_generator.generate_hex_grid(
                self.miso_footprint, 
                self.hex_size_km
            )
            
            # Validate grid
            if not self.grid_generator.validate_grid():
                raise ValueError("Grid validation failed")
            
            # Log summary
            self._log_spatial_summary()
            
            logging.info("✓ Spatial framework initialization complete")
            
            return self.miso_footprint, self.hex_grid
            
        except Exception as e:
            logging.error(f"Spatial framework initialization failed: {e}")
            raise
    
    def _log_spatial_summary(self) -> None:
        """Log summary of spatial framework"""
        footprint_area = self.footprint_manager.get_area_km2()
        grid_summary = self.grid_generator.get_grid_summary()
        
        logging.info("=== Spatial Framework Summary ===")
        logging.info(f"MISO footprint area: {footprint_area:,.0f} km²")
        logging.info(f"Grid cells generated: {grid_summary['n_cells']:,}")
        logging.info(f"Target hex size: {grid_summary['target_hex_size_km']} km")
        logging.info(f"Actual hex area: {grid_summary['actual_area_km2']['mean']:.1f} ± {grid_summary['actual_area_km2']['std']:.1f} km²")
        logging.info(f"Grid coverage: {(grid_summary['n_cells'] * grid_summary['actual_area_km2']['mean'] / footprint_area * 100):.1f}%")
    
    def spatial_aggregate_to_grid(self, data: gpd.GeoDataFrame, 
                                 value_column: str,
                                 agg_method: str = "mean") -> pd.DataFrame:
        """Aggregate point/polygon data to grid cells
        
        Args:
            data: GeoDataFrame with data to aggregate
            value_column: Column name to aggregate
            agg_method: Aggregation method ('mean', 'sum', 'max', 'min', 'count')
            
        Returns:
            DataFrame with aggregated values by cell_id
        """
        if self.hex_grid is None:
            raise ValueError("Grid not initialized. Call initialize_spatial_framework() first.")
        
        try:
            # Ensure same CRS
            if data.crs != self.hex_grid.crs:
                data = data.to_crs(self.hex_grid.crs)
            
            # Spatial join
            joined = gpd.sjoin(data, self.hex_grid, how='inner', predicate='intersects')
            
            # Aggregate by cell_id
            if agg_method == "mean":
                result = joined.groupby('cell_id')[value_column].mean()
            elif agg_method == "sum":
                result = joined.groupby('cell_id')[value_column].sum()
            elif agg_method == "max":
                result = joined.groupby('cell_id')[value_column].max()
            elif agg_method == "min":
                result = joined.groupby('cell_id')[value_column].min()
            elif agg_method == "count":
                result = joined.groupby('cell_id')[value_column].count()
            else:
                raise ValueError(f"Unknown aggregation method: {agg_method}")
            
            # Convert to DataFrame with cell_id as column
            result_df = result.reset_index()
            result_df.columns = ['cell_id', f'{value_column}_{agg_method}']
            
            return result_df
            
        except Exception as e:
            logging.error(f"Spatial aggregation failed: {e}")
            raise
    
    def get_footprint(self) -> gpd.GeoDataFrame:
        """Get MISO footprint GeoDataFrame"""
        if self.miso_footprint is None:
            raise ValueError("Footprint not initialized")
        return self.miso_footprint.copy()
    
    def get_grid(self) -> gpd.GeoDataFrame:
        """Get hexagonal grid GeoDataFrame"""
        if self.hex_grid is None:
            raise ValueError("Grid not initialized")
        return self.hex_grid.copy()
    
    def export_spatial_data(self, output_dir: str = "data/processed") -> None:
        """Export footprint and grid to files"""
        if self.miso_footprint is not None:
            footprint_path = os.path.join(output_dir, "miso_footprint.geojson")
            self.miso_footprint.to_file(footprint_path, driver="GeoJSON")
            logging.info(f"Footprint exported to: {footprint_path}")
        
        if self.hex_grid is not None:
            grid_path = os.path.join(output_dir, "miso_hex_grid.geojson")
            self.hex_grid.to_file(grid_path, driver="GeoJSON")
            logging.info(f"Grid exported to: {grid_path}")


class SpatialProcessingTests:
    """Unit tests for spatial accuracy and coverage validation"""
    
    def __init__(self, spatial_engine: SpatialProcessingEngine):
        self.spatial_engine = spatial_engine
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all spatial processing tests"""
        logging.info("Running spatial processing tests...")
        
        tests = [
            ('test_footprint_geometry', self.test_footprint_geometry),
            ('test_footprint_area', self.test_footprint_area),
            ('test_grid_generation', self.test_grid_generation),
            ('test_grid_coverage', self.test_grid_coverage),
            ('test_cell_id_uniqueness', self.test_cell_id_uniqueness),
            ('test_centroid_calculation', self.test_centroid_calculation),
            ('test_spatial_aggregation', self.test_spatial_aggregation),
            ('test_coordinate_system', self.test_coordinate_system)
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
        logging.info(f"Test Results: {passed}/{total} passed")
        
        return self.test_results
    
    def test_footprint_geometry(self) -> bool:
        """Test footprint geometry validity"""
        footprint = self.spatial_engine.get_footprint()
        
        # Check geometry validity
        if not footprint.geometry.is_valid.all():
            return False
        
        # Check it's a polygon/multipolygon
        geom_types = footprint.geometry.geom_type.unique()
        valid_types = {'Polygon', 'MultiPolygon'}
        if not set(geom_types).issubset(valid_types):
            return False
        
        # Check bounds are reasonable (roughly North America)
        minx, miny, maxx, maxy = footprint.total_bounds
        if not (-110 < minx < -80 and 25 < miny < 50 and -100 < maxx < -75 and 30 < maxy < 55):
            return False
        
        return True
    
    def test_footprint_area(self) -> bool:
        """Test footprint area is reasonable"""
        area_km2 = self.spatial_engine.footprint_manager.get_area_km2()
        
        # MISO footprint should be roughly 1.5-3 million km²
        return 1e6 < area_km2 < 5e6
    
    def test_grid_generation(self) -> bool:
        """Test grid generation produces valid hexagons"""
        grid = self.spatial_engine.get_grid()
        
        # Check all geometries are valid
        if not grid.geometry.is_valid.all():
            return False
        
        # Check all are polygons
        if not (grid.geometry.geom_type == 'Polygon').all():
            return False
        
        # Check reasonable number of cells
        n_cells = len(grid)
        if not (50 < n_cells < 10000):
            return False
        
        # Check hexagons have ~6 vertices (allowing for some variation)
        vertex_counts = [len(geom.exterior.coords) - 1 for geom in grid.geometry]
        avg_vertices = np.mean(vertex_counts)
        if not (5 < avg_vertices < 8):  # Should be close to 6
            return False
        
        return True
    
    def test_grid_coverage(self) -> bool:
        """Test grid covers footprint adequately"""
        footprint_area = self.spatial_engine.footprint_manager.get_area_km2()
        grid = self.spatial_engine.get_grid()
        grid_total_area = grid['area_km2'].sum()
        coverage_ratio = grid_total_area / footprint_area
        
        # Coverage should be reasonable (70-120% due to clipping effects)
        return 0.7 < coverage_ratio < 1.2
    
    def test_cell_id_uniqueness(self) -> bool:
        """Test all cell IDs are unique"""
        grid = self.spatial_engine.get_grid()
        
        # Check no duplicate cell IDs
        return not grid['cell_id'].duplicated().any()
    
    def test_centroid_calculation(self) -> bool:
        """Test centroid calculations are accurate"""
        grid = self.spatial_engine.get_grid()
        
        # Calculate centroids manually and compare
        manual_centroids = grid.geometry.centroid
        
        # Check longitude values match
        lon_diff = np.abs(grid['centroid_lon'] - manual_centroids.x)
        if lon_diff.max() > 1e-6:  # Very small tolerance
            return False
        
        # Check latitude values match
        lat_diff = np.abs(grid['centroid_lat'] - manual_centroids.y)
        if lat_diff.max() > 1e-6:
            return False
        
        # Check centroids are within footprint bounds
        footprint_bounds = self.spatial_engine.get_footprint().total_bounds
        minx, miny, maxx, maxy = footprint_bounds
        
        if not (grid['centroid_lon'].between(minx, maxx).all() and 
                grid['centroid_lat'].between(miny, maxy).all()):
            return False
        
        return True
    
    def test_spatial_aggregation(self) -> bool:
        """Test spatial aggregation functionality"""
        # Create test point data
        grid = self.spatial_engine.get_grid()
        
        # Generate random points within grid bounds
        minx, miny, maxx, maxy = grid.total_bounds
        n_points = 100
        
        np.random.seed(42)  # Reproducible test
        test_points = gpd.GeoDataFrame({
            'value': np.random.uniform(0, 100, n_points),
            'geometry': [Point(x, y) for x, y in 
                        zip(np.random.uniform(minx, maxx, n_points),
                            np.random.uniform(miny, maxy, n_points))]
        }, crs=grid.crs)
        
        # Test aggregation
        try:
            result = self.spatial_engine.spatial_aggregate_to_grid(
                test_points, 'value', 'mean'
            )
            
            # Check result structure
            if not isinstance(result, pd.DataFrame):
                return False
            
            if 'cell_id' not in result.columns or 'value_mean' not in result.columns:
                return False
            
            # Check values are reasonable
            if result['value_mean'].isna().all():
                return False
            
            return True
            
        except Exception:
            return False
    
    def test_coordinate_system(self) -> bool:
        """Test coordinate system consistency"""
        footprint = self.spatial_engine.get_footprint()
        grid = self.spatial_engine.get_grid()
        
        expected_crs = self.spatial_engine.crs
        
        # Check CRS matches expected
        if str(footprint.crs) != expected_crs:
            return False
        
        if str(grid.crs) != expected_crs:
            return False
        
        return True
    
    def get_test_summary(self) -> str:
        """Get formatted test summary"""
        if not self.test_results:
            return "No tests run yet"
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        summary = f"\n=== Spatial Processing Test Summary ===\n"
        summary += f"Tests passed: {passed}/{total}\n\n"
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            summary += f"{status}: {test_name}\n"
        
        return summary