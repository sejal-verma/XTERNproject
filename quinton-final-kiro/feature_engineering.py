"""
Feature Engineering and Normalization System for MISO Weather-Stress Heatmap

This module implements the feature engineering pipeline that transforms raw weather
and infrastructure data into normalized risk factors for grid stress assessment.

Key Components:
- Thermal stress calculation (heat and cold stress)
- Wind stress calculation (gusts and sustained winds)
- Precipitation stress calculation (rain, snow, ice)
- Storm proxy calculation (combined conditions)
- Normalization and validation functions

All functions follow the transparent scoring methodology defined in the requirements:
- Risk = zscore(α×Hazard + β×Exposure + γ×Vulnerability)
- Hazard = weighted combination of thermal, wind, precipitation, and storm stress
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass


@dataclass
class ThermalThresholds:
    """Thermal stress thresholds in Fahrenheit"""
    heat_low: float = 85.0    # Heat stress starts (0 score)
    heat_high: float = 100.0  # Maximum heat stress (1 score)
    cold_low: float = 10.0    # Cold stress starts (0 score)
    cold_high: float = 0.0    # Maximum cold stress (1 score)


@dataclass
class WindThresholds:
    """Wind stress thresholds in mph"""
    gust_low: float = 20.0    # Wind stress starts (0 score)
    gust_high: float = 50.0   # Maximum wind stress (1 score)
    sustained_threshold: float = 30.0  # Sustained wind bonus threshold


@dataclass
class PrecipThresholds:
    """Precipitation stress thresholds"""
    rain_heavy: float = 10.0  # Heavy rain threshold (mm/h)
    snow_heavy: float = 5.0   # Heavy snow threshold (cm/h)
    ice_threshold: float = 0.0  # Any ice = maximum score


class FeatureEngineeringEngine:
    """
    Main feature engineering engine that transforms raw weather and infrastructure
    data into normalized risk factors for grid stress assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineering engine with configuration.
        
        Args:
            config: Configuration dictionary with thresholds and weights
        """
        self.config = config or self._get_default_config()
        self.thermal_thresholds = ThermalThresholds(**self.config['thresholds']['thermal'])
        self.wind_thresholds = WindThresholds(**self.config['thresholds']['wind'])
        self.precip_thresholds = PrecipThresholds(**self.config['thresholds']['precip'])
        
        logging.info("Feature engineering engine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided"""
        return {
            'thresholds': {
                'thermal': {
                    'heat_low': 85.0,
                    'heat_high': 100.0,
                    'cold_low': 10.0,
                    'cold_high': 0.0
                },
                'wind': {
                    'gust_low': 20.0,
                    'gust_high': 50.0,
                    'sustained_threshold': 30.0
                },
                'precip': {
                    'rain_heavy': 10.0,
                    'snow_heavy': 5.0,
                    'ice_threshold': 0.0
                }
            }
        }
    
    def calculate_thermal_stress(self, 
                               temperature_f: Union[float, np.ndarray, pd.Series],
                               heat_index_f: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate thermal stress score combining heat and cold stress.
        
        Heat stress: 0 at HI≤85°F, 1 at HI≥100°F, linear interpolation
        Cold stress: 0 at T≥10°F, 1 at T≤0°F, linear interpolation
        Final score: max(heat_score, cold_score)
        
        Args:
            temperature_f: Temperature in Fahrenheit
            heat_index_f: Heat index in Fahrenheit
            
        Returns:
            Thermal stress score [0, 1]
        """
        # Calculate heat stress using heat index
        heat_stress = self._linear_interpolation(
            heat_index_f,
            self.thermal_thresholds.heat_low,
            self.thermal_thresholds.heat_high
        )
        
        # Calculate cold stress using temperature
        cold_stress = self._reverse_linear_interpolation(
            temperature_f,
            self.thermal_thresholds.cold_high,
            self.thermal_thresholds.cold_low
        )
        
        # Return maximum of heat and cold stress
        if isinstance(temperature_f, (np.ndarray, pd.Series)):
            return np.maximum(heat_stress, cold_stress)
        else:
            return max(heat_stress, cold_stress)
    
    def calculate_wind_stress(self,
                            wind_speed_mph: Union[float, np.ndarray, pd.Series],
                            wind_gust_mph: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate wind stress score from wind gusts and sustained winds.
        
        Base score: 0 at gust≤20mph, 1 at gust≥50mph, linear interpolation
        Sustained wind bonus: +0.2 if sustained wind≥30mph
        Maximum score capped at 1.0
        
        Args:
            wind_speed_mph: Sustained wind speed in mph
            wind_gust_mph: Wind gust speed in mph
            
        Returns:
            Wind stress score [0, 1]
        """
        # Base wind stress from gusts
        base_stress = self._linear_interpolation(
            wind_gust_mph,
            self.wind_thresholds.gust_low,
            self.wind_thresholds.gust_high
        )
        
        # Sustained wind bonus
        if isinstance(wind_speed_mph, (np.ndarray, pd.Series)):
            sustained_bonus = np.where(
                wind_speed_mph >= self.wind_thresholds.sustained_threshold,
                0.2,
                0.0
            )
            total_stress = base_stress + sustained_bonus
            return np.minimum(total_stress, 1.0)  # Cap at 1.0
        else:
            sustained_bonus = 0.2 if wind_speed_mph >= self.wind_thresholds.sustained_threshold else 0.0
            total_stress = base_stress + sustained_bonus
            return min(total_stress, 1.0)  # Cap at 1.0
    
    def calculate_precipitation_stress(self,
                                     rain_rate_mmh: Union[float, np.ndarray, pd.Series],
                                     snow_rate_cmh: Union[float, np.ndarray, pd.Series],
                                     ice_rate_mmh: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate precipitation stress score from rain, snow, and ice.
        
        Rain stress: 0 at 0mm/h, 1 at ≥10mm/h, linear interpolation
        Snow stress: 0 at 0cm/h, 1 at ≥5cm/h, linear interpolation
        Ice stress: 1.0 for any ice accumulation
        Final score: max(rain_stress, snow_stress, ice_stress)
        
        Args:
            rain_rate_mmh: Rain rate in mm/h
            snow_rate_cmh: Snow rate in cm/h
            ice_rate_mmh: Ice accumulation rate in mm/h
            
        Returns:
            Precipitation stress score [0, 1]
        """
        # Rain stress
        rain_stress = self._linear_interpolation(
            rain_rate_mmh,
            0.0,
            self.precip_thresholds.rain_heavy
        )
        
        # Snow stress
        snow_stress = self._linear_interpolation(
            snow_rate_cmh,
            0.0,
            self.precip_thresholds.snow_heavy
        )
        
        # Ice stress - any ice = maximum score
        if isinstance(ice_rate_mmh, (np.ndarray, pd.Series)):
            ice_stress = np.where(ice_rate_mmh > self.precip_thresholds.ice_threshold, 1.0, 0.0)
            return np.maximum.reduce([rain_stress, snow_stress, ice_stress])
        else:
            ice_stress = 1.0 if ice_rate_mmh > self.precip_thresholds.ice_threshold else 0.0
            return max(rain_stress, snow_stress, ice_stress)
    
    def calculate_storm_proxy(self,
                            precipitation_rate: Union[float, np.ndarray, pd.Series],
                            wind_gust_mph: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate storm proxy score from combined precipitation and wind conditions.
        
        Storm conditions: precipitation > 0 AND wind gust ≥35mph = 1.0
        Scaled scoring: precipitation × wind gust product for partial conditions
        
        Args:
            precipitation_rate: Any precipitation rate (rain + snow + ice)
            wind_gust_mph: Wind gust speed in mph
            
        Returns:
            Storm proxy score [0, 1]
        """
        storm_wind_threshold = 35.0  # mph
        
        if isinstance(precipitation_rate, (np.ndarray, pd.Series)):
            # Full storm conditions
            full_storm = (precipitation_rate > 0) & (wind_gust_mph >= storm_wind_threshold)
            
            # Scaled scoring for partial conditions
            precip_factor = np.minimum(precipitation_rate / 5.0, 1.0)  # Normalize precip
            wind_factor = np.minimum(wind_gust_mph / 50.0, 1.0)  # Normalize wind
            scaled_score = precip_factor * wind_factor
            
            # Return full storm score (1.0) or scaled score
            return np.where(full_storm, 1.0, scaled_score)
        else:
            # Full storm conditions
            if precipitation_rate > 0 and wind_gust_mph >= storm_wind_threshold:
                return 1.0
            
            # Scaled scoring for partial conditions
            precip_factor = min(precipitation_rate / 5.0, 1.0)  # Normalize precip
            wind_factor = min(wind_gust_mph / 50.0, 1.0)  # Normalize wind
            return precip_factor * wind_factor
    
    def _linear_interpolation(self,
                            value: Union[float, np.ndarray, pd.Series],
                            low_threshold: float,
                            high_threshold: float) -> Union[float, np.ndarray, pd.Series]:
        """
        Linear interpolation between low and high thresholds.
        
        Args:
            value: Input value(s)
            low_threshold: Lower threshold (score = 0)
            high_threshold: Upper threshold (score = 1)
            
        Returns:
            Interpolated score [0, 1]
        """
        if isinstance(value, (np.ndarray, pd.Series)):
            return np.clip((value - low_threshold) / (high_threshold - low_threshold), 0.0, 1.0)
        else:
            return max(0.0, min(1.0, (value - low_threshold) / (high_threshold - low_threshold)))
    
    def _reverse_linear_interpolation(self,
                                    value: Union[float, np.ndarray, pd.Series],
                                    low_threshold: float,
                                    high_threshold: float) -> Union[float, np.ndarray, pd.Series]:
        """
        Reverse linear interpolation (higher values give lower scores).
        
        Args:
            value: Input value(s)
            low_threshold: Lower threshold (score = 1)
            high_threshold: Upper threshold (score = 0)
            
        Returns:
            Interpolated score [0, 1]
        """
        if isinstance(value, (np.ndarray, pd.Series)):
            return np.clip((high_threshold - value) / (high_threshold - low_threshold), 0.0, 1.0)
        else:
            return max(0.0, min(1.0, (high_threshold - value) / (high_threshold - low_threshold)))
    
    def process_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process complete weather dataset to extract all stress features.
        
        Args:
            weather_data: DataFrame with weather parameters
            
        Returns:
            DataFrame with added stress score columns
        """
        result_df = weather_data.copy()
        
        # Calculate thermal stress
        result_df['thermal_stress'] = self.calculate_thermal_stress(
            result_df['temp_2m'],
            result_df['heat_index']
        )
        
        # Calculate wind stress
        result_df['wind_stress'] = self.calculate_wind_stress(
            result_df['wind_speed'],
            result_df['wind_gust']
        )
        
        # Calculate total precipitation rate
        total_precip = (
            result_df.get('precip_rate', 0) +
            result_df.get('snow_rate', 0) +
            result_df.get('ice_rate', 0)
        )
        
        # Calculate precipitation stress
        result_df['precip_stress'] = self.calculate_precipitation_stress(
            result_df.get('precip_rate', 0),
            result_df.get('snow_rate', 0),
            result_df.get('ice_rate', 0)
        )
        
        # Calculate storm proxy
        result_df['storm_proxy'] = self.calculate_storm_proxy(
            total_precip,
            result_df['wind_gust']
        )
        
        logging.info(f"Processed weather features for {len(result_df)} records")
        return result_df
    
    def validate_stress_scores(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all stress scores are within [0, 1] range.
        
        Args:
            data: DataFrame with stress score columns
            
        Returns:
            Dictionary of validation results
        """
        stress_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        validation_results = {}
        
        for column in stress_columns:
            if column in data.columns:
                values = data[column]
                is_valid = (values >= 0.0).all() and (values <= 1.0).all()
                validation_results[column] = is_valid
                
                if not is_valid:
                    logging.warning(f"Invalid {column} values detected: min={values.min():.3f}, max={values.max():.3f}")
            else:
                validation_results[column] = False
                logging.warning(f"Missing stress column: {column}")
        
        return validation_results
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all stress features.
        
        Args:
            data: DataFrame with stress score columns
            
        Returns:
            Dictionary of summary statistics
        """
        stress_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        summary = {}
        
        for column in stress_columns:
            if column in data.columns:
                values = data[column]
                summary[column] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'q75': float(values.quantile(0.75)),
                    'q95': float(values.quantile(0.95))
                }
        
        return summary


def celsius_to_fahrenheit(celsius: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert temperature from Celsius to Fahrenheit"""
    return celsius * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(fahrenheit: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert temperature from Fahrenheit to Celsius"""
    return (fahrenheit - 32.0) * 5.0 / 9.0


def mph_to_ms(mph: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert wind speed from mph to m/s"""
    return mph * 0.44704


def ms_to_mph(ms: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert wind speed from m/s to mph"""
    return ms / 0.44704