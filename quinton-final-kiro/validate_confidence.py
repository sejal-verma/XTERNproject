#!/usr/bin/env python3
"""
Validation script for confidence assessment system.

This script tests the confidence assessment functionality independently
to ensure it meets the requirements.
"""

import numpy as np
import pandas as pd
import sys
import logging
from risk_scoring_engine import RiskScoringEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_confidence_basic_functionality():
    """Test basic confidence calculation functionality"""
    print("Testing basic confidence functionality...")
    
    engine = RiskScoringEngine()
    
    # Test 1: Perfect coverage, short horizon should give high confidence
    confidence_high = engine.calculate_confidence(1.0, 12)
    print(f"Perfect coverage, 12h horizon: {confidence_high:.3f}")
    assert confidence_high > 0.8, f"Expected high confidence, got {confidence_high}"
    
    # Test 2: Poor coverage should reduce confidence
    confidence_poor = engine.calculate_confidence(0.5, 12)
    print(f"Poor coverage (0.5), 12h horizon: {confidence_poor:.3f}")
    assert confidence_poor < confidence_high, "Poor coverage should reduce confidence"
    
    # Test 3: Longer horizon should reduce confidence
    confidence_long = engine.calculate_confidence(1.0, 48)
    print(f"Perfect coverage, 48h horizon: {confidence_long:.3f}")
    assert confidence_long < confidence_high, "Longer horizon should reduce confidence"
    
    # Test 4: Array inputs
    coverage_array = np.array([1.0, 0.8, 0.5])
    horizon_array = np.array([12, 24, 48])
    confidence_array = engine.calculate_confidence(coverage_array, horizon_array)
    print(f"Array inputs: {confidence_array}")
    assert len(confidence_array) == 3, "Array output should match input length"
    assert np.all(confidence_array >= 0.0) and np.all(confidence_array <= 1.0), "All confidence values should be in [0,1]"
    
    print("✓ Basic confidence functionality tests passed\n")

def test_data_coverage_calculation():
    """Test data coverage calculation"""
    print("Testing data coverage calculation...")
    
    engine = RiskScoringEngine()
    
    # Create test data with known coverage
    weather_data = pd.DataFrame({
        'cell_id': ['A', 'B', 'C'],
        'thermal_stress': [0.5, np.nan, 0.8],  # 2/3 coverage
        'wind_stress': [0.3, 0.6, 0.9],        # 3/3 coverage
        'precip_stress': [0.1, 0.2, np.nan],   # 2/3 coverage
        'storm_proxy': [0.0, 0.4, 0.7]         # 3/3 coverage
    })
    
    infrastructure_data = pd.DataFrame({
        'cell_id': ['A', 'B', 'C'],
        'normalized_pop_density': [0.2, 0.6, 0.9],  # 3/3 coverage
        'renewable_share': [0.3, np.nan, 0.8],      # 2/3 coverage
    })
    
    coverage_metrics = engine.calculate_data_coverage(weather_data, infrastructure_data)
    
    print(f"Weather coverage: {coverage_metrics['weather_coverage']:.3f}")
    print(f"Infrastructure coverage: {coverage_metrics['infrastructure_coverage']:.3f}")
    print(f"Overall coverage: {coverage_metrics['overall_coverage']:.3f}")
    
    # Weather coverage should be (2+3+2+3)/4/3 = 10/12 ≈ 0.833
    expected_weather = (2/3 + 3/3 + 2/3 + 3/3) / 4
    assert abs(coverage_metrics['weather_coverage'] - expected_weather) < 0.1, "Weather coverage calculation incorrect"
    
    assert 0.0 <= coverage_metrics['overall_coverage'] <= 1.0, "Overall coverage should be in [0,1]"
    
    print("✓ Data coverage calculation tests passed\n")

def test_confidence_with_horizons():
    """Test confidence decreases with forecast horizon"""
    print("Testing confidence decrease with forecast horizon...")
    
    engine = RiskScoringEngine()
    
    horizons = [12, 24, 36, 48]
    confidences = []
    
    for horizon in horizons:
        confidence = engine.calculate_confidence(0.9, horizon)  # Good coverage
        confidences.append(confidence)
        print(f"Horizon {horizon}h: confidence = {confidence:.3f}")
    
    # Check that confidence generally decreases with horizon
    for i in range(1, len(confidences)):
        assert confidences[i] <= confidences[i-1], f"Confidence should decrease with horizon: {confidences[i]} > {confidences[i-1]} at horizon {horizons[i]}"
    
    print("✓ Confidence horizon decrease tests passed\n")

def test_complete_integration():
    """Test complete integration with risk assessment"""
    print("Testing complete integration...")
    
    engine = RiskScoringEngine()
    
    # Create comprehensive test data
    weather_data = pd.DataFrame({
        'cell_id': ['A', 'B', 'C', 'A', 'B', 'C'],
        'horizon_h': [12, 12, 12, 24, 24, 24],
        'thermal_stress': [0.2, 0.6, 0.9, 0.3, 0.7, 1.0],
        'wind_stress': [0.1, 0.5, 0.8, 0.2, 0.6, 0.9],
        'precip_stress': [0.0, 0.3, 0.7, 0.1, 0.4, 0.8],
        'storm_proxy': [0.0, 0.2, 0.6, 0.1, 0.3, 0.7]
    })
    
    infrastructure_data = pd.DataFrame({
        'cell_id': ['A', 'B', 'C'],
        'normalized_pop_density': [0.2, 0.6, 0.9],
        'renewable_share': [0.3, 0.7, 0.9],
        'transmission_scarcity': [0.4, 0.5, 0.6]
    })
    
    # Create complete risk assessment with confidence
    result = engine.create_complete_risk_assessment(weather_data, infrastructure_data)
    
    print(f"Risk assessment created with {len(result)} records")
    print(f"Columns: {list(result.columns)}")
    
    # Check that confidence column exists
    assert 'confidence' in result.columns, "Confidence column should be present"
    
    # Check confidence ranges
    confidence_values = result['confidence']
    assert (confidence_values >= 0.0).all(), "All confidence values should be >= 0"
    assert (confidence_values <= 1.0).all(), "All confidence values should be <= 1"
    
    # Check confidence by horizon
    confidence_by_horizon = result.groupby('horizon_h')['confidence'].mean()
    print("Mean confidence by horizon:")
    for horizon, conf in confidence_by_horizon.items():
        print(f"  {horizon}h: {conf:.3f}")
    
    # Confidence should generally decrease with horizon
    horizons = sorted(confidence_by_horizon.index)
    for i in range(1, len(horizons)):
        current_conf = confidence_by_horizon[horizons[i]]
        prev_conf = confidence_by_horizon[horizons[i-1]]
        assert current_conf <= prev_conf + 0.01, f"Confidence should not increase significantly with horizon: {current_conf} > {prev_conf}"
    
    print("✓ Complete integration tests passed\n")

def test_validation_functions():
    """Test confidence validation functions"""
    print("Testing confidence validation functions...")
    
    engine = RiskScoringEngine()
    
    # Test with valid data
    valid_data = pd.DataFrame({
        'confidence': [0.9, 0.7, 0.5],
        'horizon_h': [12, 24, 48]
    })
    
    validation = engine.validate_confidence_ranges(valid_data)
    print(f"Validation results: {validation}")
    
    assert validation['confidence_exists'], "Should detect confidence column"
    assert validation['confidence_range_valid'], "Should validate range"
    assert validation['confidence_no_nan'], "Should detect no NaN values"
    assert validation['confidence_decreases_with_horizon'], "Should detect decreasing trend"
    
    # Test summary statistics
    summary = engine.get_confidence_summary_statistics(valid_data)
    print(f"Summary statistics: {summary}")
    
    assert 'mean' in summary, "Should include mean"
    assert 'by_horizon' in summary, "Should include horizon breakdown"
    
    print("✓ Validation function tests passed\n")

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("CONFIDENCE ASSESSMENT SYSTEM VALIDATION")
    print("=" * 60)
    
    try:
        test_confidence_basic_functionality()
        test_data_coverage_calculation()
        test_confidence_with_horizons()
        test_complete_integration()
        test_validation_functions()
        
        print("=" * 60)
        print("✅ ALL CONFIDENCE ASSESSMENT TESTS PASSED!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())