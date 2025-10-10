# Infrastructure and Exposure Data System
# Implementation for Task 4: Infrastructure Data Processing

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from abc import ABC, abstractmethod
import requests
import json
from pathlib import Path


class InfrastructureAdapter(ABC):
    """Base class for infrastructure data adapters"""
    
    @abstractmethod
    def load_data(self) -> gpd.GeoDataFrame:
        """Load infrastructure data"""
        pass
    
    @abstractmethod
    def process_to_grid(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Process data and aggregate to grid cells"""
        pass


class GenerationCapacityProcessor(InfrastructureAdapter):
    """Processor for EIA-860/923 generation capacity data"""
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.capacity_data = None
        self.fuel_type_mapping = {
            'wind': ['Wind'],
            'solar': ['Solar Photovoltaic', 'Solar Thermal'],
            'thermal': ['Natural Gas', 'Coal', 'Nuclear', 'Oil', 'Gas'],
            'hydro': ['Hydro', 'Hydroelectric'],
            'other': ['Geothermal', 'Biomass', 'Other']
        }
        
    def load_data(self) -> gpd.GeoDataFrame:
        """Load EIA capacity data or create demo data"""
        try:
            if self.demo_mode:
                logging.info("Loading demo generation capacity data...")
                self.capacity_data = self._create_demo_capacity_data()
            else:
                logging.info("Loading EIA-860/923 capacity data...")
                self.capacity_data = self._load_eia_data()
            
            logging.info(f"Loaded {len(self.capacity_data)} generation facilities")
            return self.capacity_data
            
        except Exception as e:
            logging.error(f"Error loading capacity data: {e}")
            # Fallback to demo data
            logging.info("Falling back to demo capacity data...")
            self.capacity_data = self._create_demo_capacity_data()
            return self.capacity_data
    
    def _create_demo_capacity_data(self) -> gpd.GeoDataFrame:
        """Create realistic demo capacity data for MISO region"""
        np.random.seed(42)  # Reproducible demo data
        
        # Define MISO region bounds (approximate)
        miso_bounds = {
            'lon_min': -98.0, 'lon_max': -82.0,
            'lat_min': 29.5, 'lat_max': 48.0
        }
        
        # Generate facilities with realistic distribution
        facilities = []
        facility_id = 1
        
        # Wind farms (concentrated in midwest/plains)
        wind_regions = [
            {'center': (-95.0, 42.0), 'radius': 3.0, 'n_facilities': 150},  # Iowa/Minnesota
            {'center': (-87.0, 41.0), 'radius': 2.0, 'n_facilities': 80},   # Illinois
            {'center': (-97.0, 35.0), 'radius': 2.5, 'n_facilities': 100},  # Oklahoma/Texas
        ]
        
        for region in wind_regions:
            for _ in range(region['n_facilities']):
                # Random location within region
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0, region['radius'])
                lon = region['center'][0] + radius * np.cos(angle)
                lat = region['center'][1] + radius * np.sin(angle)
                
                # Ensure within MISO bounds
                lon = np.clip(lon, miso_bounds['lon_min'], miso_bounds['lon_max'])
                lat = np.clip(lat, miso_bounds['lat_min'], miso_bounds['lat_max'])
                
                facilities.append({
                    'facility_id': f"WIND_{facility_id:04d}",
                    'facility_name': f"Wind Farm {facility_id}",
                    'fuel_type': 'Wind',
                    'capacity_mw': np.random.uniform(50, 300),
                    'longitude': lon,
                    'latitude': lat,
                    'state': self._get_demo_state(lon, lat),
                    'operational_year': np.random.randint(2005, 2024)
                })
                facility_id += 1
        
        # Solar facilities (distributed, more in southern regions)
        solar_regions = [
            {'center': (-92.0, 35.0), 'radius': 4.0, 'n_facilities': 120},  # Arkansas/Louisiana
            {'center': (-87.0, 39.0), 'radius': 3.0, 'n_facilities': 100},  # Illinois/Indiana
            {'center': (-95.0, 32.0), 'radius': 2.0, 'n_facilities': 80},   # Texas
        ]
        
        for region in solar_regions:
            for _ in range(region['n_facilities']):
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0, region['radius'])
                lon = region['center'][0] + radius * np.cos(angle)
                lat = region['center'][1] + radius * np.sin(angle)
                
                lon = np.clip(lon, miso_bounds['lon_min'], miso_bounds['lon_max'])
                lat = np.clip(lat, miso_bounds['lat_min'], miso_bounds['lat_max'])
                
                facilities.append({
                    'facility_id': f"SOLAR_{facility_id:04d}",
                    'facility_name': f"Solar Farm {facility_id}",
                    'fuel_type': 'Solar Photovoltaic',
                    'capacity_mw': np.random.uniform(20, 200),
                    'longitude': lon,
                    'latitude': lat,
                    'state': self._get_demo_state(lon, lat),
                    'operational_year': np.random.randint(2010, 2024)
                })
                facility_id += 1
        
        # Natural gas plants (near population centers)
        gas_locations = [
            (-87.6, 41.9),  # Chicago area
            (-90.2, 38.6),  # St. Louis area
            (-86.1, 39.8),  # Indianapolis area
            (-94.6, 39.1),  # Kansas City area
            (-93.3, 44.9),  # Minneapolis area
            (-92.3, 34.7),  # Little Rock area
            (-90.1, 29.9),  # New Orleans area
        ]
        
        for i, (lon, lat) in enumerate(gas_locations):
            # Multiple plants per major city
            for j in range(np.random.randint(3, 8)):
                # Scatter around city center
                offset_lon = np.random.uniform(-0.5, 0.5)
                offset_lat = np.random.uniform(-0.5, 0.5)
                
                facilities.append({
                    'facility_id': f"GAS_{facility_id:04d}",
                    'facility_name': f"Natural Gas Plant {facility_id}",
                    'fuel_type': 'Natural Gas',
                    'capacity_mw': np.random.uniform(200, 1500),
                    'longitude': lon + offset_lon,
                    'latitude': lat + offset_lat,
                    'state': self._get_demo_state(lon, lat),
                    'operational_year': np.random.randint(1990, 2020)
                })
                facility_id += 1
        
        # Coal plants (fewer, larger, older)
        coal_locations = [
            (-87.0, 41.0),  # Illinois
            (-86.0, 40.0),  # Indiana
            (-94.0, 39.0),  # Missouri
            (-93.0, 45.0),  # Minnesota
            (-92.0, 35.0),  # Arkansas
        ]
        
        for lon, lat in coal_locations:
            for _ in range(np.random.randint(2, 5)):
                offset_lon = np.random.uniform(-1.0, 1.0)
                offset_lat = np.random.uniform(-1.0, 1.0)
                
                facilities.append({
                    'facility_id': f"COAL_{facility_id:04d}",
                    'facility_name': f"Coal Plant {facility_id}",
                    'fuel_type': 'Coal',
                    'capacity_mw': np.random.uniform(500, 2000),
                    'longitude': lon + offset_lon,
                    'latitude': lat + offset_lat,
                    'state': self._get_demo_state(lon, lat),
                    'operational_year': np.random.randint(1970, 2010)
                })
                facility_id += 1
        
        # Nuclear plants (few, very large)
        nuclear_locations = [
            (-87.8, 41.2),  # Illinois (multiple plants)
            (-86.5, 41.7),  # Michigan
            (-93.2, 45.1),  # Minnesota
            (-92.6, 38.7),  # Missouri
        ]
        
        for lon, lat in nuclear_locations:
            facilities.append({
                'facility_id': f"NUCLEAR_{facility_id:04d}",
                'facility_name': f"Nuclear Plant {facility_id}",
                'fuel_type': 'Nuclear',
                'capacity_mw': np.random.uniform(1000, 3000),
                'longitude': lon,
                'latitude': lat,
                'state': self._get_demo_state(lon, lat),
                'operational_year': np.random.randint(1975, 1995)
            })
            facility_id += 1
        
        # Create GeoDataFrame
        df = pd.DataFrame(facilities)
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        logging.info(f"Created demo capacity data: {len(gdf)} facilities")
        logging.info(f"Fuel mix: {gdf['fuel_type'].value_counts().to_dict()}")
        logging.info(f"Total capacity: {gdf['capacity_mw'].sum():.0f} MW")
        
        return gdf
    
    def _get_demo_state(self, lon: float, lat: float) -> str:
        """Simple state assignment based on coordinates"""
        # Very rough state boundaries for demo purposes
        if lon > -85 and lat > 41:
            return 'MI'
        elif lon > -87 and lat > 39:
            return 'IN'
        elif lon > -91 and lat > 40:
            return 'IL'
        elif lon > -94 and lat > 43:
            return 'WI'
        elif lon > -97 and lat > 43:
            return 'MN'
        elif lon > -96 and lat > 40:
            return 'IA'
        elif lon > -95 and lat < 37:
            return 'TX'
        elif lat < 33:
            return 'LA'
        elif lat < 36:
            return 'AR'
        else:
            return 'MO'
    
    def _load_eia_data(self) -> gpd.GeoDataFrame:
        """Load actual EIA-860/923 data (placeholder for real implementation)"""
        # This would implement actual EIA API calls or file loading
        # For now, fallback to demo data
        logging.warning("EIA data loading not implemented, using demo data")
        return self._create_demo_capacity_data()
    
    def process_to_grid(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate renewable share and capacity metrics for each grid cell"""
        if self.capacity_data is None:
            raise ValueError("Capacity data not loaded. Call load_data() first.")
        
        try:
            logging.info("Processing capacity data to grid...")
            
            # Ensure same CRS
            if self.capacity_data.crs != grid.crs:
                capacity_data = self.capacity_data.to_crs(grid.crs)
            else:
                capacity_data = self.capacity_data.copy()
            
            # Create buffer around each grid cell (50km as specified in requirements)
            grid_utm = grid.to_crs('EPSG:3857')  # UTM for accurate distance
            grid_buffered = grid_utm.copy()
            grid_buffered['geometry'] = grid_buffered.geometry.buffer(50000)  # 50km buffer
            grid_buffered = grid_buffered.to_crs(grid.crs)  # Back to original CRS
            
            # Spatial join capacity data with buffered grid
            capacity_utm = capacity_data.to_crs('EPSG:3857')
            grid_buffered_utm = grid_buffered.to_crs('EPSG:3857')
            
            joined = gpd.sjoin(capacity_utm, grid_buffered_utm, how='inner', predicate='intersects')
            
            # Calculate metrics for each cell
            results = []
            
            for cell_id in grid['cell_id']:
                cell_facilities = joined[joined['cell_id'] == cell_id]
                
                if len(cell_facilities) == 0:
                    # No facilities within 50km
                    results.append({
                        'cell_id': cell_id,
                        'total_capacity_mw': 0.0,
                        'renewable_capacity_mw': 0.0,
                        'thermal_capacity_mw': 0.0,
                        'renewable_share': 0.0,
                        'capacity_density_mw_km2': 0.0,
                        'facility_count': 0,
                        'wind_capacity_mw': 0.0,
                        'solar_capacity_mw': 0.0,
                        'gas_capacity_mw': 0.0,
                        'coal_capacity_mw': 0.0,
                        'nuclear_capacity_mw': 0.0
                    })
                    continue
                
                # Calculate capacity by fuel type
                fuel_capacities = cell_facilities.groupby('fuel_type')['capacity_mw'].sum()
                
                # Categorize fuels
                renewable_fuels = ['Wind', 'Solar Photovoltaic', 'Solar Thermal']
                thermal_fuels = ['Natural Gas', 'Coal', 'Nuclear', 'Oil', 'Gas']
                
                renewable_capacity = fuel_capacities[fuel_capacities.index.isin(renewable_fuels)].sum()
                thermal_capacity = fuel_capacities[fuel_capacities.index.isin(thermal_fuels)].sum()
                total_capacity = cell_facilities['capacity_mw'].sum()
                
                # Calculate renewable share
                renewable_share = renewable_capacity / total_capacity if total_capacity > 0 else 0.0
                
                # Get cell area for density calculation
                cell_area_km2 = grid[grid['cell_id'] == cell_id]['area_km2'].iloc[0]
                capacity_density = total_capacity / cell_area_km2 if cell_area_km2 > 0 else 0.0
                
                # Individual fuel type capacities
                wind_capacity = fuel_capacities.get('Wind', 0.0)
                solar_capacity = fuel_capacities.get('Solar Photovoltaic', 0.0) + fuel_capacities.get('Solar Thermal', 0.0)
                gas_capacity = fuel_capacities.get('Natural Gas', 0.0) + fuel_capacities.get('Gas', 0.0)
                coal_capacity = fuel_capacities.get('Coal', 0.0)
                nuclear_capacity = fuel_capacities.get('Nuclear', 0.0)
                
                results.append({
                    'cell_id': cell_id,
                    'total_capacity_mw': total_capacity,
                    'renewable_capacity_mw': renewable_capacity,
                    'thermal_capacity_mw': thermal_capacity,
                    'renewable_share': renewable_share,
                    'capacity_density_mw_km2': capacity_density,
                    'facility_count': len(cell_facilities),
                    'wind_capacity_mw': wind_capacity,
                    'solar_capacity_mw': solar_capacity,
                    'gas_capacity_mw': gas_capacity,
                    'coal_capacity_mw': coal_capacity,
                    'nuclear_capacity_mw': nuclear_capacity
                })
            
            result_df = pd.DataFrame(results)
            
            logging.info(f"Processed capacity data for {len(result_df)} grid cells")
            logging.info(f"Cells with capacity data: {(result_df['total_capacity_mw'] > 0).sum()}")
            logging.info(f"Average renewable share: {result_df['renewable_share'].mean():.2f}")
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error processing capacity data to grid: {e}")
            raise
    
    def get_capacity_summary(self) -> Dict:
        """Get summary statistics for capacity data"""
        if self.capacity_data is None:
            return {}
        
        fuel_summary = self.capacity_data.groupby('fuel_type')['capacity_mw'].agg(['count', 'sum', 'mean'])
        
        return {
            'total_facilities': len(self.capacity_data),
            'total_capacity_mw': self.capacity_data['capacity_mw'].sum(),
            'fuel_mix': fuel_summary.to_dict(),
            'geographic_bounds': tuple(self.capacity_data.total_bounds)
        }


class PopulationExposureProcessor(InfrastructureAdapter):
    """Processor for Census population density and load exposure data"""
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.population_data = None
        self.load_centers = None
        
    def load_data(self) -> gpd.GeoDataFrame:
        """Load Census population data or create demo data"""
        try:
            if self.demo_mode:
                logging.info("Loading demo population density data...")
                self.population_data = self._create_demo_population_data()
            else:
                logging.info("Loading Census population data...")
                self.population_data = self._load_census_data()
            
            # Load major load centers
            self.load_centers = self._create_load_centers()
            
            logging.info(f"Loaded population data for {len(self.population_data)} areas")
            return self.population_data
            
        except Exception as e:
            logging.error(f"Error loading population data: {e}")
            # Fallback to demo data
            logging.info("Falling back to demo population data...")
            self.population_data = self._create_demo_population_data()
            self.load_centers = self._create_load_centers()
            return self.population_data
    
    def _create_demo_population_data(self) -> gpd.GeoDataFrame:
        """Create realistic demo population density data"""
        np.random.seed(42)
        
        # Major metropolitan areas in MISO region with approximate populations
        metro_areas = [
            {'name': 'Chicago', 'center': (-87.6, 41.9), 'population': 9500000, 'radius': 1.5},
            {'name': 'Detroit', 'center': (-83.0, 42.3), 'population': 4300000, 'radius': 1.2},
            {'name': 'Minneapolis', 'center': (-93.3, 44.9), 'population': 3600000, 'radius': 1.0},
            {'name': 'St. Louis', 'center': (-90.2, 38.6), 'population': 2800000, 'radius': 0.8},
            {'name': 'Kansas City', 'center': (-94.6, 39.1), 'population': 2200000, 'radius': 0.7},
            {'name': 'Indianapolis', 'center': (-86.1, 39.8), 'population': 2000000, 'radius': 0.6},
            {'name': 'New Orleans', 'center': (-90.1, 29.9), 'population': 1300000, 'radius': 0.5},
            {'name': 'Milwaukee', 'center': (-87.9, 43.0), 'population': 1600000, 'radius': 0.5},
            {'name': 'Little Rock', 'center': (-92.3, 34.7), 'population': 700000, 'radius': 0.4},
            {'name': 'Des Moines', 'center': (-93.6, 41.6), 'population': 700000, 'radius': 0.4},
        ]
        
        population_areas = []
        area_id = 1
        
        # Create population density areas around metro centers
        for metro in metro_areas:
            center_lon, center_lat = metro['center']
            base_population = metro['population']
            max_radius = metro['radius']
            
            # Create concentric rings with decreasing density
            rings = [
                {'radius': max_radius * 0.3, 'density_factor': 1.0},    # Urban core
                {'radius': max_radius * 0.6, 'density_factor': 0.6},   # Suburban
                {'radius': max_radius * 1.0, 'density_factor': 0.3},   # Exurban
                {'radius': max_radius * 1.5, 'density_factor': 0.1},   # Rural fringe
            ]
            
            for i, ring in enumerate(rings):
                # Create circular area for this ring
                inner_radius = rings[i-1]['radius'] if i > 0 else 0
                outer_radius = ring['radius']
                
                # Approximate area calculation (degrees to km²)
                ring_area_km2 = np.pi * (outer_radius**2 - inner_radius**2) * (111**2)  # rough conversion
                
                # Population in this ring
                ring_population = base_population * ring['density_factor'] * 0.25  # 25% per ring
                population_density = ring_population / ring_area_km2 if ring_area_km2 > 0 else 0
                
                # Create polygon for ring (simplified as circle)
                angles = np.linspace(0, 2*np.pi, 32)
                outer_coords = [(center_lon + outer_radius * np.cos(a), 
                               center_lat + outer_radius * np.sin(a)) for a in angles]
                
                if inner_radius > 0:
                    inner_coords = [(center_lon + inner_radius * np.cos(a), 
                                   center_lat + inner_radius * np.sin(a)) for a in angles[::-1]]
                    # Create ring polygon (outer - inner)
                    ring_polygon = Polygon(outer_coords, [inner_coords])
                else:
                    # Solid circle for center
                    ring_polygon = Polygon(outer_coords)
                
                population_areas.append({
                    'area_id': f"POP_{area_id:04d}",
                    'metro_name': metro['name'],
                    'ring_type': ['urban_core', 'suburban', 'exurban', 'rural_fringe'][i],
                    'population': ring_population,
                    'area_km2': ring_area_km2,
                    'population_density_per_km2': population_density,
                    'geometry': ring_polygon
                })
                area_id += 1
        
        # Add rural background population
        rural_areas = []
        miso_bounds = (-98.0, 29.5, -82.0, 48.0)  # minx, miny, maxx, maxy
        
        # Create grid of rural areas
        rural_grid_size = 1.0  # 1 degree grid
        for lon in np.arange(miso_bounds[0], miso_bounds[2], rural_grid_size):
            for lat in np.arange(miso_bounds[1], miso_bounds[3], rural_grid_size):
                # Skip if too close to metro areas
                too_close = False
                for metro in metro_areas:
                    dist = np.sqrt((lon - metro['center'][0])**2 + (lat - metro['center'][1])**2)
                    if dist < metro['radius'] * 2:  # Skip if within 2x metro radius
                        too_close = True
                        break
                
                if not too_close:
                    # Create rural square
                    rural_polygon = Polygon([
                        (lon, lat), (lon + rural_grid_size, lat),
                        (lon + rural_grid_size, lat + rural_grid_size),
                        (lon, lat + rural_grid_size), (lon, lat)
                    ])
                    
                    rural_area_km2 = (rural_grid_size * 111)**2  # Approximate
                    rural_population = np.random.uniform(5000, 50000)  # Rural population
                    rural_density = rural_population / rural_area_km2
                    
                    rural_areas.append({
                        'area_id': f"RURAL_{area_id:04d}",
                        'metro_name': 'Rural',
                        'ring_type': 'rural',
                        'population': rural_population,
                        'area_km2': rural_area_km2,
                        'population_density_per_km2': rural_density,
                        'geometry': rural_polygon
                    })
                    area_id += 1
        
        # Combine metro and rural areas
        all_areas = population_areas + rural_areas
        
        # Create GeoDataFrame
        df = pd.DataFrame(all_areas)
        gdf = gpd.GeoDataFrame(df, crs='EPSG:4326')
        
        logging.info(f"Created demo population data: {len(gdf)} areas")
        logging.info(f"Total population: {gdf['population'].sum():,.0f}")
        logging.info(f"Density range: {gdf['population_density_per_km2'].min():.1f} - {gdf['population_density_per_km2'].max():.1f} per km²")
        
        return gdf
    
    def _create_load_centers(self) -> gpd.GeoDataFrame:
        """Create major load centers for optional load factor weighting"""
        load_centers = [
            {'name': 'Chicago Loop', 'location': (-87.6, 41.9), 'load_factor': 1.0},
            {'name': 'Detroit Downtown', 'location': (-83.0, 42.3), 'load_factor': 0.8},
            {'name': 'Minneapolis Downtown', 'location': (-93.3, 44.9), 'load_factor': 0.7},
            {'name': 'St. Louis Downtown', 'location': (-90.2, 38.6), 'load_factor': 0.6},
            {'name': 'Indianapolis Downtown', 'location': (-86.1, 39.8), 'load_factor': 0.5},
            {'name': 'Kansas City Downtown', 'location': (-94.6, 39.1), 'load_factor': 0.5},
            {'name': 'New Orleans CBD', 'location': (-90.1, 29.9), 'load_factor': 0.4},
        ]
        
        data = []
        for center in load_centers:
            data.append({
                'load_center_name': center['name'],
                'load_factor': center['load_factor'],
                'geometry': Point(center['location'])
            })
        
        return gpd.GeoDataFrame(data, crs='EPSG:4326')
    
    def _load_census_data(self) -> gpd.GeoDataFrame:
        """Load actual Census data (placeholder for real implementation)"""
        # This would implement actual Census API calls or file loading
        logging.warning("Census data loading not implemented, using demo data")
        return self._create_demo_population_data()
    
    def process_to_grid(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Aggregate population data to grid cells and calculate exposure metrics"""
        if self.population_data is None:
            raise ValueError("Population data not loaded. Call load_data() first.")
        
        try:
            logging.info("Processing population data to grid...")
            
            # Ensure same CRS
            if self.population_data.crs != grid.crs:
                population_data = self.population_data.to_crs(grid.crs)
            else:
                population_data = self.population_data.copy()
            
            # Spatial intersection to calculate population per grid cell
            results = []
            
            for _, cell in grid.iterrows():
                cell_id = cell['cell_id']
                cell_geom = cell['geometry']
                cell_area_km2 = cell['area_km2']
                
                # Find intersecting population areas
                intersecting = population_data[population_data.geometry.intersects(cell_geom)]
                
                if len(intersecting) == 0:
                    # No population data for this cell
                    results.append({
                        'cell_id': cell_id,
                        'total_population': 0.0,
                        'population_density_per_km2': 0.0,
                        'normalized_pop_density': 0.0,
                        'load_factor': 0.0,
                        'exposure_score': 0.0
                    })
                    continue
                
                # Calculate population in this cell
                total_population = 0.0
                
                for _, pop_area in intersecting.iterrows():
                    # Calculate intersection area
                    intersection = cell_geom.intersection(pop_area['geometry'])
                    
                    if intersection.is_empty:
                        continue
                    
                    # Calculate intersection area in km²
                    intersection_gdf = gpd.GeoDataFrame([1], geometry=[intersection], crs=grid.crs)
                    intersection_utm = intersection_gdf.to_crs('EPSG:3857')
                    intersection_area_km2 = intersection_utm.geometry.area.iloc[0] / 1e6
                    
                    # Proportion of population area in this cell
                    area_proportion = intersection_area_km2 / pop_area['area_km2'] if pop_area['area_km2'] > 0 else 0
                    
                    # Add proportional population
                    total_population += pop_area['population'] * area_proportion
                
                # Calculate density
                population_density = total_population / cell_area_km2 if cell_area_km2 > 0 else 0
                
                # Calculate load factor (distance to nearest major load center)
                load_factor = self._calculate_load_factor(cell['centroid_lon'], cell['centroid_lat'])
                
                results.append({
                    'cell_id': cell_id,
                    'total_population': total_population,
                    'population_density_per_km2': population_density,
                    'load_factor': load_factor
                })
            
            result_df = pd.DataFrame(results)
            
            # Normalize population density to [0,1] scale
            max_density = result_df['population_density_per_km2'].max()
            if max_density > 0:
                result_df['normalized_pop_density'] = result_df['population_density_per_km2'] / max_density
            else:
                result_df['normalized_pop_density'] = 0.0
            
            # Calculate exposure score (weighted combination of population and load factor)
            pop_weight = self.config.get('weights', {}).get('exposure', {}).get('pop', 0.7)
            load_weight = self.config.get('weights', {}).get('exposure', {}).get('load', 0.3)
            
            result_df['exposure_score'] = (
                pop_weight * result_df['normalized_pop_density'] + 
                load_weight * result_df['load_factor']
            )
            
            logging.info(f"Processed population data for {len(result_df)} grid cells")
            logging.info(f"Population range: {result_df['total_population'].min():.0f} - {result_df['total_population'].max():.0f}")
            logging.info(f"Density range: {result_df['population_density_per_km2'].min():.1f} - {result_df['population_density_per_km2'].max():.1f} per km²")
            logging.info(f"Average exposure score: {result_df['exposure_score'].mean():.3f}")
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error processing population data to grid: {e}")
            raise
    
    def _calculate_load_factor(self, lon: float, lat: float) -> float:
        """Calculate load factor based on distance to major load centers"""
        if self.load_centers is None or len(self.load_centers) == 0:
            return 0.0
        
        # Calculate distance to nearest load center
        point = Point(lon, lat)
        distances = []
        
        for _, center in self.load_centers.iterrows():
            # Simple Euclidean distance (rough approximation)
            dist = point.distance(center['geometry'])
            # Weight by load center importance
            weighted_dist = dist / center['load_factor'] if center['load_factor'] > 0 else float('inf')
            distances.append(weighted_dist)
        
        if not distances:
            return 0.0
        
        # Convert distance to load factor (closer = higher factor)
        min_distance = min(distances)
        max_distance = 5.0  # 5 degrees as max distance for load factor calculation
        
        # Inverse distance relationship, normalized to [0,1]
        load_factor = max(0.0, 1.0 - (min_distance / max_distance))
        
        return load_factor
    
    def get_population_summary(self) -> Dict:
        """Get summary statistics for population data"""
        if self.population_data is None:
            return {}
        
        return {
            'total_areas': len(self.population_data),
            'total_population': self.population_data['population'].sum(),
            'density_stats': {
                'mean': self.population_data['population_density_per_km2'].mean(),
                'std': self.population_data['population_density_per_km2'].std(),
                'min': self.population_data['population_density_per_km2'].min(),
                'max': self.population_data['population_density_per_km2'].max()
            },
            'geographic_bounds': tuple(self.population_data.total_bounds)
        }


class TransmissionDensityProcessor(InfrastructureAdapter):
    """Processor for transmission line density data"""
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.transmission_data = None
        self.baseline_scarcity = 0.5  # Default baseline as specified in requirements
        
    def load_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load transmission line data or return None if unavailable"""
        try:
            if self.demo_mode:
                logging.info("Loading demo transmission line data...")
                self.transmission_data = self._create_demo_transmission_data()
            else:
                logging.info("Attempting to load public transmission line data...")
                self.transmission_data = self._load_public_transmission_data()
            
            if self.transmission_data is not None:
                logging.info(f"Loaded {len(self.transmission_data)} transmission lines")
            else:
                logging.info("No transmission data available, will use baseline values")
            
            return self.transmission_data
            
        except Exception as e:
            logging.error(f"Error loading transmission data: {e}")
            logging.info("Falling back to baseline transmission scarcity values")
            self.transmission_data = None
            return None
    
    def _create_demo_transmission_data(self) -> gpd.GeoDataFrame:
        """Create realistic demo transmission line data"""
        np.random.seed(42)
        
        # Major transmission corridors in MISO region
        corridors = [
            # North-South corridors
            {'name': 'Eastern Interconnect', 'path': [(-87.0, 48.0), (-87.0, 30.0)], 'voltage': 765},
            {'name': 'Central Corridor', 'path': [(-93.0, 47.0), (-93.0, 32.0)], 'voltage': 500},
            {'name': 'Western Tie', 'path': [(-97.0, 45.0), (-97.0, 33.0)], 'voltage': 345},
            
            # East-West corridors  
            {'name': 'Northern Tie', 'path': [(-98.0, 45.0), (-83.0, 45.0)], 'voltage': 500},
            {'name': 'Central Cross', 'path': [(-96.0, 40.0), (-84.0, 40.0)], 'voltage': 345},
            {'name': 'Southern Grid', 'path': [(-95.0, 33.0), (-88.0, 33.0)], 'voltage': 500},
            
            # Metropolitan area networks
            {'name': 'Chicago Network', 'path': [(-88.5, 42.5), (-86.5, 41.5)], 'voltage': 345},
            {'name': 'Twin Cities Network', 'path': [(-94.0, 45.5), (-92.5, 44.5)], 'voltage': 230},
            {'name': 'St. Louis Network', 'path': [(-91.0, 39.0), (-89.5, 38.0)], 'voltage': 230},
        ]
        
        transmission_lines = []
        line_id = 1
        
        for corridor in corridors:
            path = corridor['path']
            voltage = corridor['voltage']
            
            # Create line segments between path points
            for i in range(len(path) - 1):
                start_point = path[i]
                end_point = path[i + 1]
                
                # Create multiple parallel lines for major corridors
                num_lines = 2 if voltage >= 500 else 1
                
                for line_num in range(num_lines):
                    # Add small offset for parallel lines
                    offset = 0.05 * line_num if num_lines > 1 else 0
                    
                    start_lon = start_point[0] + offset
                    start_lat = start_point[1] + offset
                    end_lon = end_point[0] + offset
                    end_lat = end_point[1] + offset
                    
                    # Create line geometry
                    from shapely.geometry import LineString
                    line_geom = LineString([(start_lon, start_lat), (end_lon, end_lat)])
                    
                    transmission_lines.append({
                        'line_id': f"TX_{line_id:04d}",
                        'line_name': f"{corridor['name']} {line_num + 1}",
                        'voltage_kv': voltage,
                        'line_type': 'AC',
                        'owner': 'Demo Utility',
                        'in_service': True,
                        'geometry': line_geom
                    })
                    line_id += 1
        
        # Add local distribution lines around major cities
        cities = [
            (-87.6, 41.9),  # Chicago
            (-93.3, 44.9),  # Minneapolis
            (-90.2, 38.6),  # St. Louis
            (-86.1, 39.8),  # Indianapolis
            (-94.6, 39.1),  # Kansas City
        ]
        
        for city_lon, city_lat in cities:
            # Create radial lines from city center
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                # Lines extending 50-100 km from city
                radius = np.random.uniform(0.5, 1.0)  # degrees (roughly 50-100 km)
                
                end_lon = city_lon + radius * np.cos(angle)
                end_lat = city_lat + radius * np.sin(angle)
                
                line_geom = LineString([(city_lon, city_lat), (end_lon, end_lat)])
                
                transmission_lines.append({
                    'line_id': f"TX_{line_id:04d}",
                    'line_name': f"Local Distribution {line_id}",
                    'voltage_kv': np.random.choice([138, 230, 345]),
                    'line_type': 'AC',
                    'owner': 'Local Utility',
                    'in_service': True,
                    'geometry': line_geom
                })
                line_id += 1
        
        # Create GeoDataFrame
        df = pd.DataFrame(transmission_lines)
        gdf = gpd.GeoDataFrame(df, crs='EPSG:4326')
        
        logging.info(f"Created demo transmission data: {len(gdf)} lines")
        voltage_summary = gdf['voltage_kv'].value_counts().sort_index()
        logging.info(f"Voltage levels: {voltage_summary.to_dict()}")
        
        return gdf
    
    def _load_public_transmission_data(self) -> Optional[gpd.GeoDataFrame]:
        """Attempt to load public transmission data (placeholder)"""
        # This would implement loading from public sources like:
        # - EIA-860 transmission data
        # - HIFLD transmission lines
        # - State utility commission data
        # For now, return None to trigger baseline fallback
        
        logging.warning("Public transmission data loading not implemented")
        return None
    
    def process_to_grid(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate transmission density metrics for each grid cell"""
        try:
            logging.info("Processing transmission density to grid...")
            
            results = []
            
            if self.transmission_data is None:
                # Use baseline scarcity value for all cells
                logging.info(f"Using baseline transmission scarcity value: {self.baseline_scarcity}")
                
                for cell_id in grid['cell_id']:
                    results.append({
                        'cell_id': cell_id,
                        'transmission_line_count': 0,
                        'transmission_length_km': 0.0,
                        'transmission_density_km_per_km2': 0.0,
                        'high_voltage_lines': 0,
                        'transmission_scarcity': self.baseline_scarcity,
                        'normalized_tx_density': self.baseline_scarcity,
                        'data_source': 'baseline'
                    })
                
                result_df = pd.DataFrame(results)
                logging.info(f"Applied baseline transmission scarcity to {len(result_df)} grid cells")
                return result_df
            
            # Process actual transmission data
            # Ensure same CRS
            if self.transmission_data.crs != grid.crs:
                transmission_data = self.transmission_data.to_crs(grid.crs)
            else:
                transmission_data = self.transmission_data.copy()
            
            for _, cell in grid.iterrows():
                cell_id = cell['cell_id']
                cell_geom = cell['geometry']
                cell_area_km2 = cell['area_km2']
                
                # Find transmission lines intersecting this cell
                intersecting_lines = transmission_data[transmission_data.geometry.intersects(cell_geom)]
                
                if len(intersecting_lines) == 0:
                    # No transmission lines in this cell
                    results.append({
                        'cell_id': cell_id,
                        'transmission_line_count': 0,
                        'transmission_length_km': 0.0,
                        'transmission_density_km_per_km2': 0.0,
                        'high_voltage_lines': 0,
                        'transmission_scarcity': 1.0,  # High scarcity (no lines)
                        'data_source': 'calculated'
                    })
                    continue
                
                # Calculate transmission metrics
                total_length_km = 0.0
                high_voltage_count = 0
                
                for _, line in intersecting_lines.iterrows():
                    # Calculate intersection length
                    intersection = cell_geom.intersection(line['geometry'])
                    
                    if intersection.is_empty:
                        continue
                    
                    # Convert to UTM for accurate length calculation
                    intersection_gdf = gpd.GeoDataFrame([1], geometry=[intersection], crs=grid.crs)
                    intersection_utm = intersection_gdf.to_crs('EPSG:3857')
                    
                    # Calculate length in km
                    if hasattr(intersection_utm.geometry.iloc[0], 'length'):
                        length_m = intersection_utm.geometry.iloc[0].length
                        total_length_km += length_m / 1000.0
                    
                    # Count high voltage lines (≥345 kV)
                    if line['voltage_kv'] >= 345:
                        high_voltage_count += 1
                
                # Calculate density
                transmission_density = total_length_km / cell_area_km2 if cell_area_km2 > 0 else 0.0
                
                # Calculate scarcity (inverse of density, normalized)
                # Higher density = lower scarcity
                max_expected_density = 2.0  # km/km² as reasonable maximum
                normalized_density = min(transmission_density / max_expected_density, 1.0)
                transmission_scarcity = 1.0 - normalized_density
                
                results.append({
                    'cell_id': cell_id,
                    'transmission_line_count': len(intersecting_lines),
                    'transmission_length_km': total_length_km,
                    'transmission_density_km_per_km2': transmission_density,
                    'high_voltage_lines': high_voltage_count,
                    'transmission_scarcity': transmission_scarcity,
                    'data_source': 'calculated'
                })
            
            result_df = pd.DataFrame(results)
            
            # Normalize transmission density to [0,1] scale
            max_density = result_df['transmission_density_km_per_km2'].max()
            if max_density > 0:
                result_df['normalized_tx_density'] = result_df['transmission_density_km_per_km2'] / max_density
            else:
                result_df['normalized_tx_density'] = 0.0
            
            logging.info(f"Processed transmission data for {len(result_df)} grid cells")
            logging.info(f"Cells with transmission lines: {(result_df['transmission_line_count'] > 0).sum()}")
            logging.info(f"Average transmission scarcity: {result_df['transmission_scarcity'].mean():.3f}")
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error processing transmission data to grid: {e}")
            raise
    
    def get_transmission_summary(self) -> Dict:
        """Get summary statistics for transmission data"""
        if self.transmission_data is None:
            return {
                'data_available': False,
                'baseline_scarcity': self.baseline_scarcity,
                'note': 'Using baseline transmission scarcity values'
            }
        
        voltage_summary = self.transmission_data['voltage_kv'].value_counts().sort_index()
        
        return {
            'data_available': True,
            'total_lines': len(self.transmission_data),
            'voltage_levels': voltage_summary.to_dict(),
            'geographic_bounds': tuple(self.transmission_data.total_bounds),
            'baseline_scarcity': self.baseline_scarcity
        }


class InfrastructureDataSystem:
    """Main interface for infrastructure and exposure data processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.demo_mode = config['runtime']['mode'] == 'demo'
        
        # Initialize processors
        self.capacity_processor = GenerationCapacityProcessor(config, self.demo_mode)
        self.population_processor = PopulationExposureProcessor(config, self.demo_mode)
        self.transmission_processor = TransmissionDensityProcessor(config, self.demo_mode)
        
        # Storage for processed results
        self.capacity_results = None
        self.population_results = None
        self.transmission_results = None
        self.combined_results = None
    
    def initialize_infrastructure_data(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Initialize complete infrastructure data system
        
        Args:
            grid: Hexagonal grid GeoDataFrame
            
        Returns:
            Combined DataFrame with all infrastructure metrics by cell_id
        """
        try:
            logging.info("Initializing infrastructure data system...")
            
            # Step 1: Load and process generation capacity data
            logging.info("Processing generation capacity data...")
            self.capacity_processor.load_data()
            self.capacity_results = self.capacity_processor.process_to_grid(grid)
            
            # Step 2: Load and process population exposure data
            logging.info("Processing population exposure data...")
            self.population_processor.load_data()
            self.population_results = self.population_processor.process_to_grid(grid)
            
            # Step 3: Load and process transmission density data
            logging.info("Processing transmission density data...")
            self.transmission_processor.load_data()
            self.transmission_results = self.transmission_processor.process_to_grid(grid)
            
            # Step 4: Combine all results
            logging.info("Combining infrastructure data...")
            self.combined_results = self._combine_infrastructure_data()
            
            # Log summary
            self._log_infrastructure_summary()
            
            logging.info("✓ Infrastructure data system initialization complete")
            
            return self.combined_results
            
        except Exception as e:
            logging.error(f"Infrastructure data system initialization failed: {e}")
            raise
    
    def _combine_infrastructure_data(self) -> pd.DataFrame:
        """Combine all infrastructure data into single DataFrame"""
        # Start with capacity data
        combined = self.capacity_results.copy()
        
        # Merge population data
        combined = combined.merge(
            self.population_results, 
            on='cell_id', 
            how='outer',
            suffixes=('', '_pop')
        )
        
        # Merge transmission data
        combined = combined.merge(
            self.transmission_results, 
            on='cell_id', 
            how='outer',
            suffixes=('', '_tx')
        )
        
        # Fill any missing values with zeros/defaults
        combined = combined.fillna(0.0)
        
        return combined
    
    def _log_infrastructure_summary(self) -> None:
        """Log summary of infrastructure data processing"""
        logging.info("=== Infrastructure Data Summary ===")
        
        # Capacity summary
        capacity_summary = self.capacity_processor.get_capacity_summary()
        if capacity_summary:
            logging.info(f"Generation facilities: {capacity_summary['total_facilities']:,}")
            logging.info(f"Total capacity: {capacity_summary['total_capacity_mw']:,.0f} MW")
        
        # Population summary
        pop_summary = self.population_processor.get_population_summary()
        if pop_summary:
            logging.info(f"Total population: {pop_summary['total_population']:,.0f}")
            logging.info(f"Population areas: {pop_summary['total_areas']:,}")
        
        # Transmission summary
        tx_summary = self.transmission_processor.get_transmission_summary()
        if tx_summary['data_available']:
            logging.info(f"Transmission lines: {tx_summary['total_lines']:,}")
        else:
            logging.info(f"Transmission: Using baseline scarcity ({tx_summary['baseline_scarcity']})")
        
        # Combined results summary
        if self.combined_results is not None:
            logging.info(f"Grid cells processed: {len(self.combined_results):,}")
            logging.info(f"Cells with capacity data: {(self.combined_results['total_capacity_mw'] > 0).sum():,}")
            logging.info(f"Cells with population: {(self.combined_results['total_population'] > 0).sum():,}")
    
    def get_infrastructure_data(self) -> pd.DataFrame:
        """Get combined infrastructure data DataFrame"""
        if self.combined_results is None:
            raise ValueError("Infrastructure data not initialized")
        return self.combined_results.copy()
    
    def export_infrastructure_data(self, output_dir: str = "data/processed") -> None:
        """Export infrastructure data to files"""
        if self.combined_results is not None:
            # Export combined results
            combined_path = os.path.join(output_dir, "infrastructure_data.csv")
            self.combined_results.to_csv(combined_path, index=False)
            logging.info(f"Infrastructure data exported to: {combined_path}")
            
            # Export individual components
            if self.capacity_results is not None:
                capacity_path = os.path.join(output_dir, "capacity_data.csv")
                self.capacity_results.to_csv(capacity_path, index=False)
                
            if self.population_results is not None:
                population_path = os.path.join(output_dir, "population_data.csv")
                self.population_results.to_csv(population_path, index=False)
                
            if self.transmission_results is not None:
                transmission_path = os.path.join(output_dir, "transmission_data.csv")
                self.transmission_results.to_csv(transmission_path, index=False)