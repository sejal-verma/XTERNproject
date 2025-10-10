#!/usr/bin/env python3
"""
Test script for MISO Spatial Framework Implementation
Task 2: MISO footprint and hexagonal grid generation
"""

import sys
import logging
from spatial_framework import (
    MISOFootprint, 
    HexGridGenerator, 
    SpatialProcessingEngine, 
    SpatialProcessingTests
)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print('🧪 Testing MISO Spatial Framework Implementation...')
    print('Task 2: MISO footprint and hexagonal grid generation')
    
    # Test configuration
    config = {
        'runtime': {
            'mode': 'demo',
            'crs': 'EPSG:4326',
            'hex_size_km': 40
        }
    }
    
    try:
        # Initialize spatial engine
        print('\n1️⃣ Initializing spatial processing engine...')
        spatial_engine = SpatialProcessingEngine(config)
        print('   ✅ Spatial engine created')
        
        # Initialize spatial framework
        print('\n2️⃣ Creating MISO footprint and hexagonal grid...')
        miso_footprint, hex_grid = spatial_engine.initialize_spatial_framework()
        
        print('   ✅ Spatial framework initialized successfully!')
        print(f'   📍 MISO footprint: {spatial_engine.footprint_manager.get_area_km2():,.0f} km²')
        print(f'   🔷 Hex grid cells: {len(hex_grid):,}')
        print(f'   📏 Average cell area: {hex_grid["area_km2"].mean():.1f} km²')
        
        # Display bounds
        minx, miny, maxx, maxy = miso_footprint.total_bounds
        print(f'   🗺️  Footprint bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})')
        
        # Run comprehensive tests
        print('\n3️⃣ Running unit tests for spatial accuracy and coverage validation...')
        spatial_tests = SpatialProcessingTests(spatial_engine)
        test_results = spatial_tests.run_all_tests()
        
        # Test summary
        passed = sum(test_results.values())
        total = len(test_results)
        print(f'\n📊 Test Results: {passed}/{total} passed')
        
        if all(test_results.values()):
            print('\n🎉 All tests PASSED! Implementation is working correctly.')
            print('   ✅ Footprint geometry is valid')
            print('   ✅ Grid generation produces proper hexagons')
            print('   ✅ Cell IDs are unique')
            print('   ✅ Centroids are calculated correctly')
            print('   ✅ Spatial aggregation works properly')
            print('   ✅ Coordinate systems are consistent')
        else:
            failed = [name for name, result in test_results.items() if not result]
            print(f'\n⚠️  Failed tests: {failed}')
            return False
        
        # Export data
        print('\n4️⃣ Exporting spatial data...')
        spatial_engine.export_spatial_data()
        print('   ✅ Exported miso_footprint.geojson')
        print('   ✅ Exported miso_hex_grid.geojson')
        
        # Final summary
        print('\n' + '='*60)
        print('🗺️  TASK 2 IMPLEMENTATION COMPLETE')
        print('   ✅ MISO footprint defined and validated')
        print('   ✅ Hexagonal grid generated with ~40km spacing')
        print('   ✅ Grid clipped to MISO footprint')
        print('   ✅ Cell IDs assigned and centroids calculated')
        print('   ✅ Unit tests for spatial accuracy passed')
        print('   ✅ Coverage validation completed')
        print('   ✅ Spatial data exported for future use')
        print('='*60)
        
        return True
        
    except Exception as e:
        print(f'\n❌ Error during implementation: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)