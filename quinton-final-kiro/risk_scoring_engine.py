"""
Risk Scoring Engine for MISO Weather-Stress Heatmap

This module implements the comprehensive risk scoring system that combines
weather hazards, infrastructure exposure, and grid vulnerability into final
risk scores with confidence metrics.

Key Components:
- Hazard score calculation (thermal, wind, precipitation, storm)
- Exposure score calculation (population density, load factors)
- Vulnerability score calculation (renewable share, transmission scarcity)
- Final risk score calculation with z-score normalization
- Confidence assessment based on data coverage and forecast horizon

Risk Formula: Risk = zscore(α×Hazard + β×Exposure + γ×Vulnerability)
Confidence Formula: Confidence = base_confidence × coverage_factor × horizon_factor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class RiskWeights:
    """Configuration for risk scoring weights"""
    # Hazard component weights
    thermal: float = 0.3
    wind: float = 0.3
    precip: float = 0.25
    storm: float = 0.15
    
    # Exposure component weights
    pop: float = 0.7
    load: float = 0.3
    
    # Vulnerability component weights
    renew_share: float = 0.6
    tx_scarcity: float = 0.3
    outage: float = 0.1
    
    # Final blend weights
    alpha: float = 0.5  # hazard weight
    beta: float = 0.3   # exposure weight
    gamma: float = 0.2  # vulnerability weight


@dataclass
class RiskAssessment:
    """Risk assessment result for a single grid cell"""
    cell_id: str
    horizon_h: int
    hazard_score: float
    exposure_score: float
    vulnerability_score: float
    final_risk: float
    confidence: float
    top_contributors: List[str]


class RiskScoringEngine:
    """
    Main risk scoring engine that combines hazard, exposure, and vulnerability
    scores into final risk assessments with confidence metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk scoring engine with configuration.
        
        Args:
            config: Configuration dictionary with weights and parameters
        """
        self.config = config or self._get_default_config()
        self.weights = RiskWeights(**self.config['weights'])
        
        # Validate weights
        self._validate_weights()
        
        logging.info("Risk scoring engine initialized")
        logging.info(f"Hazard weights: T={self.weights.thermal}, W={self.weights.wind}, "
                    f"P={self.weights.precip}, S={self.weights.storm}")
        logging.info(f"Exposure weights: Pop={self.weights.pop}, Load={self.weights.load}")
        logging.info(f"Vulnerability weights: Ren={self.weights.renew_share}, "
                    f"Tx={self.weights.tx_scarcity}, Out={self.weights.outage}")
        logging.info(f"Blend weights: α={self.weights.alpha}, β={self.weights.beta}, "
                    f"γ={self.weights.gamma}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided"""
        return {
            'weights': {
                'thermal': 0.3,
                'wind': 0.3,
                'precip': 0.25,
                'storm': 0.15,
                'pop': 0.7,
                'load': 0.3,
                'renew_share': 0.6,
                'tx_scarcity': 0.3,
                'outage': 0.1,
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2
            }
        }
    
    def _validate_weights(self):
        """Validate that weights are properly configured"""
        # Check hazard weights sum to 1.0
        hazard_sum = (self.weights.thermal + self.weights.wind + 
                     self.weights.precip + self.weights.storm)
        if not np.isclose(hazard_sum, 1.0, atol=1e-3):
            logging.warning(f"Hazard weights sum to {hazard_sum:.3f}, not 1.0")
        
        # Check exposure weights sum to 1.0
        exposure_sum = self.weights.pop + self.weights.load
        if not np.isclose(exposure_sum, 1.0, atol=1e-3):
            logging.warning(f"Exposure weights sum to {exposure_sum:.3f}, not 1.0")
        
        # Check vulnerability weights sum to 1.0
        vulnerability_sum = (self.weights.renew_share + self.weights.tx_scarcity + 
                           self.weights.outage)
        if not np.isclose(vulnerability_sum, 1.0, atol=1e-3):
            logging.warning(f"Vulnerability weights sum to {vulnerability_sum:.3f}, not 1.0")
        
        # Check blend weights sum to 1.0
        blend_sum = self.weights.alpha + self.weights.beta + self.weights.gamma
        if not np.isclose(blend_sum, 1.0, atol=1e-3):
            logging.warning(f"Blend weights sum to {blend_sum:.3f}, not 1.0")
        
        # Check all weights are non-negative
        weight_dict = {
            'thermal': self.weights.thermal,
            'wind': self.weights.wind,
            'precip': self.weights.precip,
            'storm': self.weights.storm,
            'pop': self.weights.pop,
            'load': self.weights.load,
            'renew_share': self.weights.renew_share,
            'tx_scarcity': self.weights.tx_scarcity,
            'outage': self.weights.outage,
            'alpha': self.weights.alpha,
            'beta': self.weights.beta,
            'gamma': self.weights.gamma
        }
        
        for name, weight in weight_dict.items():
            if weight < 0:
                raise ValueError(f"Weight '{name}' cannot be negative: {weight}")
    
    def _validate_score_range(self, 
                             score: Union[float, np.ndarray, pd.Series], 
                             name: str):
        """Validate that score is in [0,1] range"""
        if isinstance(score, (np.ndarray, pd.Series)):
            if np.any(score < 0) or np.any(score > 1):
                min_val = np.min(score)
                max_val = np.max(score)
                logging.warning(f"{name} has values outside [0,1]: min={min_val:.3f}, max={max_val:.3f}")
        else:
            if score < 0 or score > 1:
                logging.warning(f"{name} has value outside [0,1]: {score:.3f}")
    
    # HAZARD SCORE CALCULATION METHODS
    
    def calculate_hazard_score(self, 
                              thermal_stress: Union[float, np.ndarray, pd.Series],
                              wind_stress: Union[float, np.ndarray, pd.Series],
                              precip_stress: Union[float, np.ndarray, pd.Series],
                              storm_proxy: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate hazard score from weather stress components.
        
        Formula: Hazard = wT×Thermal + wW×Wind + wP×Precip + wS×Storm
        
        Args:
            thermal_stress: Thermal stress score [0,1]
            wind_stress: Wind stress score [0,1]
            precip_stress: Precipitation stress score [0,1]
            storm_proxy: Storm proxy score [0,1]
            
        Returns:
            Hazard score [0,1]
        """
        # Validate input ranges
        self._validate_score_range(thermal_stress, "thermal_stress")
        self._validate_score_range(wind_stress, "wind_stress")
        self._validate_score_range(precip_stress, "precip_stress")
        self._validate_score_range(storm_proxy, "storm_proxy")
        
        # Calculate weighted sum
        hazard_score = (
            self.weights.thermal * thermal_stress +
            self.weights.wind * wind_stress +
            self.weights.precip * precip_stress +
            self.weights.storm * storm_proxy
        )
        
        # Ensure result is in [0,1] range
        if isinstance(hazard_score, (np.ndarray, pd.Series)):
            hazard_score = np.clip(hazard_score, 0.0, 1.0)
        else:
            hazard_score = max(0.0, min(1.0, hazard_score))
        
        return hazard_score
    
    def process_hazard_scores(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process complete weather dataset to calculate hazard scores.
        
        Args:
            weather_data: DataFrame with stress score columns
            
        Returns:
            DataFrame with added hazard_score column
        """
        required_columns = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in weather_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = weather_data.copy()
        
        # Calculate hazard scores
        result_df['hazard_score'] = self.calculate_hazard_score(
            result_df['thermal_stress'],
            result_df['wind_stress'],
            result_df['precip_stress'],
            result_df['storm_proxy']
        )
        
        logging.info(f"Calculated hazard scores for {len(result_df)} records")
        logging.info(f"Hazard score range: {result_df['hazard_score'].min():.3f} - {result_df['hazard_score'].max():.3f}")
        logging.info(f"Mean hazard score: {result_df['hazard_score'].mean():.3f}")
        
        return result_df
    
    def get_hazard_component_contributions(self, 
                                         thermal_stress: float,
                                         wind_stress: float,
                                         precip_stress: float,
                                         storm_proxy: float) -> Dict[str, float]:
        """
        Get individual component contributions to hazard score.
        
        Args:
            thermal_stress: Thermal stress score [0,1]
            wind_stress: Wind stress score [0,1]
            precip_stress: Precipitation stress score [0,1]
            storm_proxy: Storm proxy score [0,1]
            
        Returns:
            Dictionary of component contributions
        """
        contributions = {
            'thermal': self.weights.thermal * thermal_stress,
            'wind': self.weights.wind * wind_stress,
            'precip': self.weights.precip * precip_stress,
            'storm': self.weights.storm * storm_proxy
        }
        
        return contributions
    
    def get_top_hazard_contributors(self, 
                                   thermal_stress: float,
                                   wind_stress: float,
                                   precip_stress: float,
                                   storm_proxy: float,
                                   n_top: int = 3) -> List[str]:
        """
        Get top N contributing factors to hazard score.
        
        Args:
            thermal_stress: Thermal stress score [0,1]
            wind_stress: Wind stress score [0,1]
            precip_stress: Precipitation stress score [0,1]
            storm_proxy: Storm proxy score [0,1]
            n_top: Number of top contributors to return
            
        Returns:
            List of top contributor names, sorted by contribution
        """
        contributions = self.get_hazard_component_contributions(
            thermal_stress, wind_stress, precip_stress, storm_proxy
        )
        
        # Sort by contribution value (descending)
        sorted_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N contributor names
        return [name for name, _ in sorted_contributors[:n_top]]
    
    def validate_hazard_calculation(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate hazard score calculations.
        
        Args:
            data: DataFrame with hazard scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'hazard_score' not in data.columns:
            validation_results['hazard_score_exists'] = False
            return validation_results
        
        hazard_scores = data['hazard_score']
        
        # Check range [0,1]
        validation_results['hazard_range_valid'] = (
            (hazard_scores >= 0.0).all() and (hazard_scores <= 1.0).all()
        )
        
        # Check for NaN values
        validation_results['hazard_no_nan'] = not hazard_scores.isna().any()
        
        # Check mathematical consistency (spot check)
        if all(col in data.columns for col in ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']):
            # Recalculate first few rows and compare
            sample_size = min(10, len(data))
            sample_data = data.head(sample_size)
            
            recalculated = self.calculate_hazard_score(
                sample_data['thermal_stress'],
                sample_data['wind_stress'],
                sample_data['precip_stress'],
                sample_data['storm_proxy']
            )
            
            original = sample_data['hazard_score']
            validation_results['hazard_calculation_consistent'] = np.allclose(
                recalculated, original, atol=1e-6
            )
        else:
            validation_results['hazard_calculation_consistent'] = False
        
        return validation_results
    
    def get_hazard_summary_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for hazard scores.
        
        Args:
            data: DataFrame with hazard scores
            
        Returns:
            Dictionary of summary statistics
        """
        if 'hazard_score' not in data.columns:
            return {}
        
        hazard_scores = data['hazard_score']
        
        return {
            'count': len(hazard_scores),
            'mean': float(hazard_scores.mean()),
            'std': float(hazard_scores.std()),
            'min': float(hazard_scores.min()),
            'max': float(hazard_scores.max()),
            'median': float(hazard_scores.median()),
            'q25': float(hazard_scores.quantile(0.25)),
            'q75': float(hazard_scores.quantile(0.75)),
            'q95': float(hazard_scores.quantile(0.95)),
            'q99': float(hazard_scores.quantile(0.99))
        } 
   # EXPOSURE SCORE CALCULATION METHODS
    
    def calculate_exposure_score(self,
                                population_density: Union[float, np.ndarray, pd.Series],
                                load_factor: Optional[Union[float, np.ndarray, pd.Series]] = None) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate exposure score from population density and optional load factors.
        
        Formula: Exposure = wPop×PopDensity + wLoad×LoadFactor
        If load_factor is None, uses population-only scoring with weight = 1.0
        
        Args:
            population_density: Normalized population density [0,1]
            load_factor: Optional normalized load factor [0,1]
            
        Returns:
            Exposure score [0,1]
        """
        # Validate input ranges
        self._validate_score_range(population_density, "population_density")
        
        if load_factor is not None:
            self._validate_score_range(load_factor, "load_factor")
            
            # Use configured weights
            exposure_score = (
                self.weights.pop * population_density +
                self.weights.load * load_factor
            )
        else:
            # Population-only scoring (load factor unavailable)
            logging.debug("Load factor not available, using population-only exposure scoring")
            exposure_score = population_density
        
        # Ensure result is in [0,1] range
        if isinstance(exposure_score, (np.ndarray, pd.Series)):
            exposure_score = np.clip(exposure_score, 0.0, 1.0)
        else:
            exposure_score = max(0.0, min(1.0, exposure_score))
        
        return exposure_score
    
    def process_exposure_scores(self, infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process complete infrastructure dataset to calculate exposure scores.
        
        Args:
            infrastructure_data: DataFrame with population and optional load data
            
        Returns:
            DataFrame with added exposure_score column
        """
        required_columns = ['normalized_pop_density']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in infrastructure_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = infrastructure_data.copy()
        
        # Check if load factor is available
        load_factor = None
        if 'load_factor' in result_df.columns:
            load_factor = result_df['load_factor']
            logging.info("Using load factor data for exposure calculation")
        else:
            logging.info("Load factor data not available, using population-only exposure scoring")
        
        # Calculate exposure scores
        result_df['exposure_score'] = self.calculate_exposure_score(
            result_df['normalized_pop_density'],
            load_factor
        )
        
        logging.info(f"Calculated exposure scores for {len(result_df)} records")
        logging.info(f"Exposure score range: {result_df['exposure_score'].min():.3f} - {result_df['exposure_score'].max():.3f}")
        logging.info(f"Mean exposure score: {result_df['exposure_score'].mean():.3f}")
        
        return result_df
    
    def get_exposure_component_contributions(self,
                                           population_density: float,
                                           load_factor: Optional[float] = None) -> Dict[str, float]:
        """
        Get individual component contributions to exposure score.
        
        Args:
            population_density: Normalized population density [0,1]
            load_factor: Optional normalized load factor [0,1]
            
        Returns:
            Dictionary of component contributions
        """
        contributions = {
            'population': self.weights.pop * population_density
        }
        
        if load_factor is not None:
            contributions['load'] = self.weights.load * load_factor
        else:
            contributions['load'] = 0.0
        
        return contributions
    
    def validate_exposure_against_urban_rural(self, 
                                            data: pd.DataFrame,
                                            urban_threshold: float = 0.7,
                                            rural_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Validate exposure scoring against expected urban vs rural patterns.
        
        Args:
            data: DataFrame with exposure scores and population data
            urban_threshold: Threshold for classifying urban areas (normalized pop density)
            rural_threshold: Threshold for classifying rural areas (normalized pop density)
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'exposure_score' not in data.columns or 'normalized_pop_density' not in data.columns:
            validation_results['data_available'] = False
            return validation_results
        
        validation_results['data_available'] = True
        
        # Classify areas
        urban_mask = data['normalized_pop_density'] >= urban_threshold
        rural_mask = data['normalized_pop_density'] <= rural_threshold
        
        urban_data = data[urban_mask]
        rural_data = data[rural_mask]
        
        if len(urban_data) > 0 and len(rural_data) > 0:
            urban_mean_exposure = urban_data['exposure_score'].mean()
            rural_mean_exposure = rural_data['exposure_score'].mean()
            
            validation_results['urban_count'] = len(urban_data)
            validation_results['rural_count'] = len(rural_data)
            validation_results['urban_mean_exposure'] = urban_mean_exposure
            validation_results['rural_mean_exposure'] = rural_mean_exposure
            
            # Urban areas should have higher exposure scores than rural areas
            validation_results['urban_higher_than_rural'] = urban_mean_exposure > rural_mean_exposure
            
            # Calculate correlation between population density and exposure
            correlation = data['normalized_pop_density'].corr(data['exposure_score'])
            validation_results['pop_exposure_correlation'] = correlation
            validation_results['positive_correlation'] = correlation > 0.5
            
        else:
            validation_results['urban_higher_than_rural'] = None
            validation_results['pop_exposure_correlation'] = None
            validation_results['positive_correlation'] = None
            
            if len(urban_data) == 0:
                logging.warning("No urban areas found for validation")
            if len(rural_data) == 0:
                logging.warning("No rural areas found for validation")
        
        return validation_results
    
    def validate_exposure_calculation(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate exposure score calculations.
        
        Args:
            data: DataFrame with exposure scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'exposure_score' not in data.columns:
            validation_results['exposure_score_exists'] = False
            return validation_results
        
        exposure_scores = data['exposure_score']
        
        # Check range [0,1]
        validation_results['exposure_range_valid'] = (
            (exposure_scores >= 0.0).all() and (exposure_scores <= 1.0).all()
        )
        
        # Check for NaN values
        validation_results['exposure_no_nan'] = not exposure_scores.isna().any()
        
        # Check mathematical consistency (spot check)
        if 'normalized_pop_density' in data.columns:
            sample_size = min(10, len(data))
            sample_data = data.head(sample_size)
            
            load_factor = sample_data.get('load_factor', None)
            
            recalculated = self.calculate_exposure_score(
                sample_data['normalized_pop_density'],
                load_factor
            )
            
            original = sample_data['exposure_score']
            validation_results['exposure_calculation_consistent'] = np.allclose(
                recalculated, original, atol=1e-6
            )
        else:
            validation_results['exposure_calculation_consistent'] = False
        
        return validation_results
    
    def get_exposure_summary_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for exposure scores.
        
        Args:
            data: DataFrame with exposure scores
            
        Returns:
            Dictionary of summary statistics
        """
        if 'exposure_score' not in data.columns:
            return {}
        
        exposure_scores = data['exposure_score']
        
        return {
            'count': len(exposure_scores),
            'mean': float(exposure_scores.mean()),
            'std': float(exposure_scores.std()),
            'min': float(exposure_scores.min()),
            'max': float(exposure_scores.max()),
            'median': float(exposure_scores.median()),
            'q25': float(exposure_scores.quantile(0.25)),
            'q75': float(exposure_scores.quantile(0.75)),
            'q95': float(exposure_scores.quantile(0.95)),
            'q99': float(exposure_scores.quantile(0.99))
        }    
# VULNERABILITY SCORE CALCULATION METHODS
    
    def calculate_vulnerability_score(self,
                                     renewable_share: Union[float, np.ndarray, pd.Series],
                                     transmission_scarcity: Union[float, np.ndarray, pd.Series],
                                     outage_flag: Union[bool, np.ndarray, pd.Series] = False) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate vulnerability score from renewable share, transmission scarcity, and outage flags.
        
        Formula: Vulnerability = wRen×RenewShare + wTx×TxScarcity + wOut×OutageFlag
        
        Args:
            renewable_share: Renewable generation share [0,1]
            transmission_scarcity: Transmission scarcity metric [0,1] (higher = more scarce)
            outage_flag: Boolean outage flag (converted to 0/1)
            
        Returns:
            Vulnerability score [0,1]
        """
        # Validate input ranges
        self._validate_score_range(renewable_share, "renewable_share")
        self._validate_score_range(transmission_scarcity, "transmission_scarcity")
        
        # Convert outage flag to numeric
        if isinstance(outage_flag, (np.ndarray, pd.Series)):
            outage_numeric = outage_flag.astype(float)
        else:
            outage_numeric = float(outage_flag)
        
        # Calculate weighted sum
        vulnerability_score = (
            self.weights.renew_share * renewable_share +
            self.weights.tx_scarcity * transmission_scarcity +
            self.weights.outage * outage_numeric
        )
        
        # Ensure result is in [0,1] range
        if isinstance(vulnerability_score, (np.ndarray, pd.Series)):
            vulnerability_score = np.clip(vulnerability_score, 0.0, 1.0)
        else:
            vulnerability_score = max(0.0, min(1.0, vulnerability_score))
        
        return vulnerability_score
    
    def process_vulnerability_scores(self, infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process complete infrastructure dataset to calculate vulnerability scores.
        
        Args:
            infrastructure_data: DataFrame with renewable share, transmission, and outage data
            
        Returns:
            DataFrame with added vulnerability_score column
        """
        required_columns = ['renewable_share']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in infrastructure_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = infrastructure_data.copy()
        
        # Handle transmission scarcity data
        if 'transmission_scarcity' not in result_df.columns:
            # Use baseline value as specified in requirements (0.5)
            baseline_tx_scarcity = 0.5
            result_df['transmission_scarcity'] = baseline_tx_scarcity
            logging.info(f"Transmission data not available, using baseline scarcity value: {baseline_tx_scarcity}")
        else:
            logging.info("Using available transmission scarcity data")
        
        # Handle outage flag data
        if 'outage_flag' not in result_df.columns:
            # Default to no outages
            result_df['outage_flag'] = False
            logging.info("Outage flag data not available, defaulting to False")
        else:
            logging.info("Using available outage flag data")
        
        # Calculate vulnerability scores
        result_df['vulnerability_score'] = self.calculate_vulnerability_score(
            result_df['renewable_share'],
            result_df['transmission_scarcity'],
            result_df['outage_flag']
        )
        
        logging.info(f"Calculated vulnerability scores for {len(result_df)} records")
        logging.info(f"Vulnerability score range: {result_df['vulnerability_score'].min():.3f} - {result_df['vulnerability_score'].max():.3f}")
        logging.info(f"Mean vulnerability score: {result_df['vulnerability_score'].mean():.3f}")
        
        return result_df
    
    def get_vulnerability_component_contributions(self,
                                                renewable_share: float,
                                                transmission_scarcity: float,
                                                outage_flag: bool = False) -> Dict[str, float]:
        """
        Get individual component contributions to vulnerability score.
        
        Args:
            renewable_share: Renewable generation share [0,1]
            transmission_scarcity: Transmission scarcity metric [0,1]
            outage_flag: Boolean outage flag
            
        Returns:
            Dictionary of component contributions
        """
        contributions = {
            'renewable_share': self.weights.renew_share * renewable_share,
            'transmission_scarcity': self.weights.tx_scarcity * transmission_scarcity,
            'outage': self.weights.outage * float(outage_flag)
        }
        
        return contributions
    
    def validate_vulnerability_edge_cases(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test vulnerability scoring edge cases and boundary conditions.
        
        Args:
            data: DataFrame with vulnerability scores and components
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        required_columns = ['vulnerability_score', 'renewable_share', 'transmission_scarcity']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_results['data_available'] = False
            validation_results['missing_columns'] = missing_columns
            return validation_results
        
        validation_results['data_available'] = True
        
        # Test case 1: High renewable share should increase vulnerability
        high_renewable_mask = data['renewable_share'] >= 0.8
        low_renewable_mask = data['renewable_share'] <= 0.2
        
        if high_renewable_mask.any() and low_renewable_mask.any():
            high_renewable_vuln = data[high_renewable_mask]['vulnerability_score'].mean()
            low_renewable_vuln = data[low_renewable_mask]['vulnerability_score'].mean()
            
            validation_results['high_renewable_more_vulnerable'] = high_renewable_vuln > low_renewable_vuln
            validation_results['high_renewable_vuln_mean'] = high_renewable_vuln
            validation_results['low_renewable_vuln_mean'] = low_renewable_vuln
        else:
            validation_results['high_renewable_more_vulnerable'] = None
        
        # Test case 2: High transmission scarcity should increase vulnerability
        high_tx_scarcity_mask = data['transmission_scarcity'] >= 0.8
        low_tx_scarcity_mask = data['transmission_scarcity'] <= 0.2
        
        if high_tx_scarcity_mask.any() and low_tx_scarcity_mask.any():
            high_tx_vuln = data[high_tx_scarcity_mask]['vulnerability_score'].mean()
            low_tx_vuln = data[low_tx_scarcity_mask]['vulnerability_score'].mean()
            
            validation_results['high_tx_scarcity_more_vulnerable'] = high_tx_vuln > low_tx_vuln
            validation_results['high_tx_vuln_mean'] = high_tx_vuln
            validation_results['low_tx_vuln_mean'] = low_tx_vuln
        else:
            validation_results['high_tx_scarcity_more_vulnerable'] = None
        
        # Test case 3: Outage flag effect (if available)
        if 'outage_flag' in data.columns:
            outage_mask = data['outage_flag'] == True
            no_outage_mask = data['outage_flag'] == False
            
            if outage_mask.any() and no_outage_mask.any():
                outage_vuln = data[outage_mask]['vulnerability_score'].mean()
                no_outage_vuln = data[no_outage_mask]['vulnerability_score'].mean()
                
                validation_results['outage_increases_vulnerability'] = outage_vuln > no_outage_vuln
                validation_results['outage_vuln_mean'] = outage_vuln
                validation_results['no_outage_vuln_mean'] = no_outage_vuln
            else:
                validation_results['outage_increases_vulnerability'] = None
        else:
            validation_results['outage_increases_vulnerability'] = None
        
        # Test case 4: Baseline transmission scarcity handling
        baseline_tx_mask = np.isclose(data['transmission_scarcity'], 0.5, atol=1e-3)
        if baseline_tx_mask.any():
            validation_results['baseline_tx_scarcity_count'] = baseline_tx_mask.sum()
            validation_results['baseline_tx_scarcity_fraction'] = baseline_tx_mask.mean()
        else:
            validation_results['baseline_tx_scarcity_count'] = 0
            validation_results['baseline_tx_scarcity_fraction'] = 0.0
        
        return validation_results
    
    def validate_vulnerability_calculation(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate vulnerability score calculations.
        
        Args:
            data: DataFrame with vulnerability scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'vulnerability_score' not in data.columns:
            validation_results['vulnerability_score_exists'] = False
            return validation_results
        
        vulnerability_scores = data['vulnerability_score']
        
        # Check range [0,1]
        validation_results['vulnerability_range_valid'] = (
            (vulnerability_scores >= 0.0).all() and (vulnerability_scores <= 1.0).all()
        )
        
        # Check for NaN values
        validation_results['vulnerability_no_nan'] = not vulnerability_scores.isna().any()
        
        # Check mathematical consistency (spot check)
        required_cols = ['renewable_share', 'transmission_scarcity']
        if all(col in data.columns for col in required_cols):
            sample_size = min(10, len(data))
            sample_data = data.head(sample_size)
            
            outage_flag = sample_data.get('outage_flag', False)
            
            recalculated = self.calculate_vulnerability_score(
                sample_data['renewable_share'],
                sample_data['transmission_scarcity'],
                outage_flag
            )
            
            original = sample_data['vulnerability_score']
            validation_results['vulnerability_calculation_consistent'] = np.allclose(
                recalculated, original, atol=1e-6
            )
        else:
            validation_results['vulnerability_calculation_consistent'] = False
        
        return validation_results
    
    def get_vulnerability_summary_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for vulnerability scores.
        
        Args:
            data: DataFrame with vulnerability scores
            
        Returns:
            Dictionary of summary statistics
        """
        if 'vulnerability_score' not in data.columns:
            return {}
        
        vulnerability_scores = data['vulnerability_score']
        
        return {
            'count': len(vulnerability_scores),
            'mean': float(vulnerability_scores.mean()),
            'std': float(vulnerability_scores.std()),
            'min': float(vulnerability_scores.min()),
            'max': float(vulnerability_scores.max()),
            'median': float(vulnerability_scores.median()),
            'q25': float(vulnerability_scores.quantile(0.25)),
            'q75': float(vulnerability_scores.quantile(0.75)),
            'q95': float(vulnerability_scores.quantile(0.95)),
            'q99': float(vulnerability_scores.quantile(0.99))
        }   
 # FINAL RISK SCORE CALCULATION METHODS
    
    def calculate_final_risk_score(self,
                                  hazard_score: Union[float, np.ndarray, pd.Series],
                                  exposure_score: Union[float, np.ndarray, pd.Series],
                                  vulnerability_score: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate final risk score using z-score normalization.
        
        Formula: Risk = zscore(α×Hazard + β×Exposure + γ×Vulnerability)
        
        Args:
            hazard_score: Hazard score [0,1]
            exposure_score: Exposure score [0,1]
            vulnerability_score: Vulnerability score [0,1]
            
        Returns:
            Final risk score (z-scored)
        """
        # Validate input ranges
        self._validate_score_range(hazard_score, "hazard_score")
        self._validate_score_range(exposure_score, "exposure_score")
        self._validate_score_range(vulnerability_score, "vulnerability_score")
        
        # Calculate weighted combination
        combined_score = (
            self.weights.alpha * hazard_score +
            self.weights.beta * exposure_score +
            self.weights.gamma * vulnerability_score
        )
        
        # Apply z-score normalization
        if isinstance(combined_score, (np.ndarray, pd.Series)):
            # Calculate z-scores across all values
            mean_score = np.mean(combined_score)
            std_score = np.std(combined_score, ddof=1)
            
            if std_score > 0:
                risk_score = (combined_score - mean_score) / std_score
            else:
                # If no variation, all scores are the same
                risk_score = np.zeros_like(combined_score)
                logging.warning("No variation in combined scores, z-scores set to 0")
        else:
            # Single value - cannot calculate z-score without population
            # Return the combined score as-is with warning
            logging.warning("Cannot calculate z-score for single value, returning combined score")
            risk_score = combined_score
        
        return risk_score
    
    def calculate_final_risk_scores_by_horizon(self, 
                                             combined_data: pd.DataFrame,
                                             horizon_column: str = 'horizon_h') -> pd.DataFrame:
        """
        Calculate final risk scores with z-score normalization by forecast horizon.
        
        Args:
            combined_data: DataFrame with hazard, exposure, vulnerability scores
            horizon_column: Column name for forecast horizon
            
        Returns:
            DataFrame with added final_risk column
        """
        required_columns = ['hazard_score', 'exposure_score', 'vulnerability_score']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = combined_data.copy()
        
        # Calculate z-scores separately for each forecast horizon
        if horizon_column in result_df.columns:
            logging.info(f"Calculating z-scores separately for each forecast horizon")
            
            # Initialize final_risk column
            result_df['final_risk'] = np.nan
            
            for horizon in result_df[horizon_column].unique():
                horizon_mask = result_df[horizon_column] == horizon
                horizon_data = result_df[horizon_mask]
                
                if len(horizon_data) == 0:
                    continue
                
                # Calculate combined scores for this horizon
                combined_scores = (
                    self.weights.alpha * horizon_data['hazard_score'] +
                    self.weights.beta * horizon_data['exposure_score'] +
                    self.weights.gamma * horizon_data['vulnerability_score']
                )
                
                # Calculate z-scores for this horizon
                mean_score = combined_scores.mean()
                std_score = combined_scores.std(ddof=1)
                
                if std_score > 0:
                    z_scores = (combined_scores - mean_score) / std_score
                else:
                    z_scores = pd.Series(0.0, index=combined_scores.index)
                    logging.warning(f"No variation in combined scores for horizon {horizon}h, z-scores set to 0")
                
                # Assign z-scores back to result dataframe
                result_df.loc[horizon_mask, 'final_risk'] = z_scores
                
                logging.info(f"Horizon {horizon}h: mean={mean_score:.3f}, std={std_score:.3f}, "
                           f"z-score range=[{z_scores.min():.3f}, {z_scores.max():.3f}]")
        
        else:
            # No horizon column - calculate z-scores across all data
            logging.info("No horizon column found, calculating z-scores across all data")
            
            result_df['final_risk'] = self.calculate_final_risk_score(
                result_df['hazard_score'],
                result_df['exposure_score'],
                result_df['vulnerability_score']
            )
        
        logging.info(f"Calculated final risk scores for {len(result_df)} records")
        logging.info(f"Final risk score range: {result_df['final_risk'].min():.3f} - {result_df['final_risk'].max():.3f}")
        logging.info(f"Mean final risk score: {result_df['final_risk'].mean():.3f}")
        
        return result_df
    
    def get_risk_component_contributions(self,
                                       hazard_score: float,
                                       exposure_score: float,
                                       vulnerability_score: float) -> Dict[str, float]:
        """
        Get individual component contributions to final risk score.
        
        Args:
            hazard_score: Hazard score [0,1]
            exposure_score: Exposure score [0,1]
            vulnerability_score: Vulnerability score [0,1]
            
        Returns:
            Dictionary of component contributions (before z-scoring)
        """
        contributions = {
            'hazard': self.weights.alpha * hazard_score,
            'exposure': self.weights.beta * exposure_score,
            'vulnerability': self.weights.gamma * vulnerability_score
        }
        
        return contributions
    
    def get_top_risk_contributors(self,
                                hazard_score: float,
                                exposure_score: float,
                                vulnerability_score: float,
                                n_top: int = 3) -> List[str]:
        """
        Get top N contributing factors to final risk score.
        
        Args:
            hazard_score: Hazard score [0,1]
            exposure_score: Exposure score [0,1]
            vulnerability_score: Vulnerability score [0,1]
            n_top: Number of top contributors to return
            
        Returns:
            List of top contributor names, sorted by contribution
        """
        contributions = self.get_risk_component_contributions(
            hazard_score, exposure_score, vulnerability_score
        )
        
        # Sort by contribution value (descending)
        sorted_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N contributor names
        return [name for name, _ in sorted_contributors[:n_top]]
    
    def validate_risk_score_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate final risk score distribution and mathematical correctness.
        
        Args:
            data: DataFrame with final risk scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'final_risk' not in data.columns:
            validation_results['final_risk_exists'] = False
            return validation_results
        
        validation_results['final_risk_exists'] = True
        risk_scores = data['final_risk']
        
        # Check for NaN values
        validation_results['risk_no_nan'] = not risk_scores.isna().any()
        
        # Check z-score properties (should have mean ≈ 0, std ≈ 1 for each horizon)
        if 'horizon_h' in data.columns:
            horizon_stats = []
            for horizon in data['horizon_h'].unique():
                horizon_data = data[data['horizon_h'] == horizon]['final_risk']
                
                if len(horizon_data) > 1:
                    horizon_mean = horizon_data.mean()
                    horizon_std = horizon_data.std(ddof=1)
                    
                    horizon_stats.append({
                        'horizon': horizon,
                        'mean': horizon_mean,
                        'std': horizon_std,
                        'mean_near_zero': abs(horizon_mean) < 1e-10,
                        'std_near_one': abs(horizon_std - 1.0) < 1e-10
                    })
            
            validation_results['horizon_stats'] = horizon_stats
            validation_results['z_score_properties_valid'] = all(
                stat['mean_near_zero'] and stat['std_near_one'] 
                for stat in horizon_stats
            )
        else:
            # Single population z-score check
            if len(risk_scores) > 1:
                mean_risk = risk_scores.mean()
                std_risk = risk_scores.std(ddof=1)
                
                validation_results['mean_risk'] = mean_risk
                validation_results['std_risk'] = std_risk
                validation_results['mean_near_zero'] = abs(mean_risk) < 1e-10
                validation_results['std_near_one'] = abs(std_risk - 1.0) < 1e-10
                validation_results['z_score_properties_valid'] = (
                    validation_results['mean_near_zero'] and 
                    validation_results['std_near_one']
                )
            else:
                validation_results['z_score_properties_valid'] = None
        
        return validation_results
    
    def validate_final_risk_calculation(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate final risk score calculations.
        
        Args:
            data: DataFrame with final risk scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'final_risk' not in data.columns:
            validation_results['final_risk_exists'] = False
            return validation_results
        
        risk_scores = data['final_risk']
        
        # Check for NaN values
        validation_results['final_risk_no_nan'] = not risk_scores.isna().any()
        
        # Check for infinite values
        validation_results['final_risk_finite'] = np.isfinite(risk_scores).all()
        
        # Check that z-scores have reasonable range (typically -3 to +3)
        validation_results['final_risk_reasonable_range'] = (
            (risk_scores >= -5.0).all() and (risk_scores <= 5.0).all()
        )
        
        return validation_results
    
    def get_final_risk_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for final risk scores.
        
        Args:
            data: DataFrame with final risk scores
            
        Returns:
            Dictionary of summary statistics
        """
        if 'final_risk' not in data.columns:
            return {}
        
        risk_scores = data['final_risk']
        
        summary = {
            'count': len(risk_scores),
            'mean': float(risk_scores.mean()),
            'std': float(risk_scores.std()),
            'min': float(risk_scores.min()),
            'max': float(risk_scores.max()),
            'median': float(risk_scores.median()),
            'q25': float(risk_scores.quantile(0.25)),
            'q75': float(risk_scores.quantile(0.75)),
            'q95': float(risk_scores.quantile(0.95)),
            'q99': float(risk_scores.quantile(0.99))
        }
        
        # Add horizon-specific statistics if available
        if 'horizon_h' in data.columns:
            horizon_summary = {}
            for horizon in sorted(data['horizon_h'].unique()):
                horizon_data = data[data['horizon_h'] == horizon]['final_risk']
                horizon_summary[f'{horizon}h'] = {
                    'count': len(horizon_data),
                    'mean': float(horizon_data.mean()),
                    'std': float(horizon_data.std()),
                    'min': float(horizon_data.min()),
                    'max': float(horizon_data.max()),
                    'q95': float(horizon_data.quantile(0.95))
                }
            
            summary['by_horizon'] = horizon_summary
        
        return summary
    
    # CONFIDENCE ASSESSMENT METHODS
    
    def calculate_confidence(self, 
                           data_coverage: Union[float, np.ndarray, pd.Series],
                           horizon_h: Union[int, np.ndarray, pd.Series],
                           base_confidence: float = 0.9,
                           coverage_threshold: float = 0.8,
                           horizon_decay_rate: float = 0.02) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate confidence score based on data coverage and forecast horizon.
        
        Confidence decreases with:
        - Lower data coverage (below threshold gets penalty)
        - Longer forecast horizons (exponential decay)
        - Missing infrastructure or weather data
        
        Formula: Confidence = base_confidence × coverage_factor × horizon_factor
        
        Args:
            data_coverage: Data coverage ratio [0,1]
            horizon_h: Forecast horizon in hours
            base_confidence: Base confidence level (default 0.9)
            coverage_threshold: Coverage threshold below which penalty applies (default 0.8)
            horizon_decay_rate: Rate of confidence decay per hour (default 0.05)
            
        Returns:
            Confidence score [0,1]
        """
        # Validate inputs
        if isinstance(data_coverage, (np.ndarray, pd.Series)):
            if np.any(data_coverage < 0) or np.any(data_coverage > 1):
                logging.warning("Data coverage values outside [0,1] range detected")
                data_coverage = np.clip(data_coverage, 0.0, 1.0)
        else:
            if data_coverage < 0 or data_coverage > 1:
                logging.warning(f"Data coverage outside [0,1] range: {data_coverage}")
                data_coverage = max(0.0, min(1.0, data_coverage))
        
        # Calculate coverage factor
        # Linear penalty below threshold, no penalty above threshold
        if isinstance(data_coverage, (np.ndarray, pd.Series)):
            coverage_factor = np.where(
                data_coverage >= coverage_threshold,
                1.0,
                data_coverage / coverage_threshold
            )
        else:
            coverage_factor = 1.0 if data_coverage >= coverage_threshold else data_coverage / coverage_threshold
        
        # Calculate horizon factor (exponential decay)
        # Confidence decreases exponentially with forecast horizon
        if isinstance(horizon_h, (np.ndarray, pd.Series)):
            horizon_factor = np.exp(-horizon_decay_rate * horizon_h)  # Direct exponential decay
        else:
            horizon_factor = np.exp(-horizon_decay_rate * horizon_h)
        
        # Calculate final confidence
        confidence = base_confidence * coverage_factor * horizon_factor
        
        # Ensure result is in [0,1] range
        if isinstance(confidence, (np.ndarray, pd.Series)):
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def calculate_data_coverage(self, 
                              weather_data: pd.DataFrame,
                              infrastructure_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data coverage metrics for confidence assessment.
        
        Args:
            weather_data: DataFrame with weather data
            infrastructure_data: DataFrame with infrastructure data
            
        Returns:
            Dictionary with coverage metrics
        """
        coverage_metrics = {}
        
        # Weather data coverage
        weather_required_cols = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
        weather_coverage = []
        
        for col in weather_required_cols:
            if col in weather_data.columns:
                # Calculate non-null coverage
                non_null_ratio = 1.0 - weather_data[col].isna().sum() / len(weather_data)
                weather_coverage.append(non_null_ratio)
            else:
                weather_coverage.append(0.0)
        
        coverage_metrics['weather_coverage'] = np.mean(weather_coverage)
        
        # Infrastructure data coverage
        infra_required_cols = ['normalized_pop_density', 'renewable_share']
        infra_optional_cols = ['load_factor', 'transmission_scarcity', 'outage_flag']
        
        infra_coverage = []
        
        # Required infrastructure data
        for col in infra_required_cols:
            if col in infrastructure_data.columns:
                non_null_ratio = 1.0 - infrastructure_data[col].isna().sum() / len(infrastructure_data)
                infra_coverage.append(non_null_ratio)
            else:
                infra_coverage.append(0.0)
        
        # Optional infrastructure data (weighted less)
        optional_coverage = []
        for col in infra_optional_cols:
            if col in infrastructure_data.columns:
                non_null_ratio = 1.0 - infrastructure_data[col].isna().sum() / len(infrastructure_data)
                optional_coverage.append(non_null_ratio)
            else:
                optional_coverage.append(0.0)  # Missing optional data gets 0
        
        # Weight required data more heavily than optional data
        required_weight = 0.8
        optional_weight = 0.2
        
        coverage_metrics['infrastructure_coverage'] = (
            required_weight * np.mean(infra_coverage) + 
            optional_weight * np.mean(optional_coverage)
        )
        
        # Overall coverage (average of weather and infrastructure)
        coverage_metrics['overall_coverage'] = (
            coverage_metrics['weather_coverage'] + 
            coverage_metrics['infrastructure_coverage']
        ) / 2.0
        
        # Individual component coverage for detailed analysis
        coverage_metrics['weather_components'] = dict(zip(weather_required_cols, weather_coverage))
        coverage_metrics['infrastructure_components'] = dict(zip(
            infra_required_cols + infra_optional_cols, 
            infra_coverage + optional_coverage
        ))
        
        return coverage_metrics
    
    def add_confidence_scores(self, 
                            risk_data: pd.DataFrame,
                            weather_data: pd.DataFrame,
                            infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add confidence scores to risk assessment data.
        
        Args:
            risk_data: DataFrame with risk scores and horizon information
            weather_data: DataFrame with weather data for coverage calculation
            infrastructure_data: DataFrame with infrastructure data for coverage calculation
            
        Returns:
            DataFrame with added confidence scores
        """
        result_df = risk_data.copy()
        
        # Calculate data coverage metrics
        coverage_metrics = self.calculate_data_coverage(weather_data, infrastructure_data)
        overall_coverage = coverage_metrics['overall_coverage']
        
        logging.info(f"Data coverage metrics: Weather={coverage_metrics['weather_coverage']:.3f}, "
                    f"Infrastructure={coverage_metrics['infrastructure_coverage']:.3f}, "
                    f"Overall={overall_coverage:.3f}")
        
        # Calculate confidence for each row based on horizon
        if 'horizon_h' in result_df.columns:
            result_df['confidence'] = self.calculate_confidence(
                overall_coverage,  # Same coverage for all cells
                result_df['horizon_h']
            )
        else:
            # If no horizon information, use default 12h horizon
            logging.warning("No horizon information found, using default 12h for confidence calculation")
            result_df['confidence'] = self.calculate_confidence(overall_coverage, 12)
        
        # Add per-cell data quality penalties if available
        if 'cell_id' in result_df.columns:
            # Check for cell-specific data quality issues
            result_df = self._apply_cell_specific_confidence_penalties(
                result_df, weather_data, infrastructure_data
            )
        
        logging.info(f"Added confidence scores: mean={result_df['confidence'].mean():.3f}, "
                    f"min={result_df['confidence'].min():.3f}, "
                    f"max={result_df['confidence'].max():.3f}")
        
        return result_df
    
    def _apply_cell_specific_confidence_penalties(self, 
                                                risk_data: pd.DataFrame,
                                                weather_data: pd.DataFrame,
                                                infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cell-specific confidence penalties based on data quality.
        
        Args:
            risk_data: DataFrame with risk scores and confidence
            weather_data: DataFrame with weather data
            infrastructure_data: DataFrame with infrastructure data
            
        Returns:
            DataFrame with adjusted confidence scores
        """
        result_df = risk_data.copy()
        
        # Create cell-level data quality metrics
        cell_quality = {}
        
        # Weather data quality by cell
        if 'cell_id' in weather_data.columns:
            weather_cols = ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']
            for cell_id in weather_data['cell_id'].unique():
                cell_weather = weather_data[weather_data['cell_id'] == cell_id]
                
                # Calculate completeness for this cell
                completeness_scores = []
                for col in weather_cols:
                    if col in cell_weather.columns:
                        completeness = 1.0 - cell_weather[col].isna().sum() / len(cell_weather)
                        completeness_scores.append(completeness)
                    else:
                        completeness_scores.append(0.0)
                
                cell_quality[cell_id] = {'weather_quality': np.mean(completeness_scores)}
        
        # Infrastructure data quality by cell
        if 'cell_id' in infrastructure_data.columns:
            infra_cols = ['normalized_pop_density', 'renewable_share', 'transmission_scarcity']
            for cell_id in infrastructure_data['cell_id'].unique():
                cell_infra = infrastructure_data[infrastructure_data['cell_id'] == cell_id]
                
                # Calculate completeness for this cell
                completeness_scores = []
                for col in infra_cols:
                    if col in cell_infra.columns:
                        completeness = 1.0 - cell_infra[col].isna().sum() / len(cell_infra)
                        completeness_scores.append(completeness)
                    else:
                        completeness_scores.append(0.0)
                
                if cell_id in cell_quality:
                    cell_quality[cell_id]['infra_quality'] = np.mean(completeness_scores)
                else:
                    cell_quality[cell_id] = {'infra_quality': np.mean(completeness_scores)}
        
        # Apply cell-specific penalties
        for idx, row in result_df.iterrows():
            cell_id = row['cell_id']
            
            if cell_id in cell_quality:
                quality_metrics = cell_quality[cell_id]
                
                # Calculate quality penalty
                weather_quality = quality_metrics.get('weather_quality', 1.0)
                infra_quality = quality_metrics.get('infra_quality', 1.0)
                
                # Average quality across data types
                cell_data_quality = (weather_quality + infra_quality) / 2.0
                
                # Apply penalty if quality is below threshold
                quality_threshold = 0.9
                if cell_data_quality < quality_threshold:
                    quality_penalty = cell_data_quality / quality_threshold
                    result_df.loc[idx, 'confidence'] *= quality_penalty
                    
                    if quality_penalty < 0.8:  # Log significant penalties
                        logging.debug(f"Applied quality penalty to cell {cell_id}: "
                                    f"penalty={quality_penalty:.3f}, "
                                    f"weather_quality={weather_quality:.3f}, "
                                    f"infra_quality={infra_quality:.3f}")
        
        # Ensure confidence remains in [0,1] range
        result_df['confidence'] = np.clip(result_df['confidence'], 0.0, 1.0)
        
        return result_df
    
    def validate_confidence_ranges(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that confidence scores are in proper [0,1] range.
        
        Args:
            data: DataFrame with confidence scores
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        if 'confidence' not in data.columns:
            validation_results['confidence_exists'] = False
            return validation_results
        
        validation_results['confidence_exists'] = True
        confidence_scores = data['confidence']
        
        # Check range [0,1]
        validation_results['confidence_range_valid'] = (
            (confidence_scores >= 0.0).all() and (confidence_scores <= 1.0).all()
        )
        
        # Check for NaN values
        validation_results['confidence_no_nan'] = not confidence_scores.isna().any()
        
        # Check that confidence decreases with horizon (if horizon data available)
        if 'horizon_h' in data.columns:
            horizon_confidence = data.groupby('horizon_h')['confidence'].mean()
            horizons = sorted(horizon_confidence.index)
            
            # Check if confidence generally decreases with horizon
            decreasing_trend = True
            for i in range(1, len(horizons)):
                if horizon_confidence[horizons[i]] > horizon_confidence[horizons[i-1]]:
                    decreasing_trend = False
                    break
            
            validation_results['confidence_decreases_with_horizon'] = decreasing_trend
        else:
            validation_results['confidence_decreases_with_horizon'] = None
        
        return validation_results
    
    def get_confidence_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for confidence scores.
        
        Args:
            data: DataFrame with confidence scores
            
        Returns:
            Dictionary of summary statistics
        """
        if 'confidence' not in data.columns:
            return {}
        
        confidence_scores = data['confidence']
        
        summary = {
            'count': len(confidence_scores),
            'mean': float(confidence_scores.mean()),
            'std': float(confidence_scores.std()),
            'min': float(confidence_scores.min()),
            'max': float(confidence_scores.max()),
            'median': float(confidence_scores.median()),
            'q25': float(confidence_scores.quantile(0.25)),
            'q75': float(confidence_scores.quantile(0.75))
        }
        
        # Add horizon-specific statistics if available
        if 'horizon_h' in data.columns:
            horizon_summary = {}
            for horizon in sorted(data['horizon_h'].unique()):
                horizon_data = data[data['horizon_h'] == horizon]['confidence']
                horizon_summary[f'{horizon}h'] = {
                    'count': len(horizon_data),
                    'mean': float(horizon_data.mean()),
                    'std': float(horizon_data.std()),
                    'min': float(horizon_data.min()),
                    'max': float(horizon_data.max())
                }
            
            summary['by_horizon'] = horizon_summary
        
        return summary
    
    def create_complete_risk_assessment(self, 
                                      weather_data: pd.DataFrame,
                                      infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete risk assessment combining all components with confidence scores.
        
        Args:
            weather_data: DataFrame with weather stress scores
            infrastructure_data: DataFrame with infrastructure scores
            
        Returns:
            DataFrame with complete risk assessment including confidence scores
        """
        # Process hazard scores
        hazard_data = self.process_hazard_scores(weather_data)
        
        # Process exposure scores
        exposure_data = self.process_exposure_scores(infrastructure_data)
        
        # Process vulnerability scores
        vulnerability_data = self.process_vulnerability_scores(infrastructure_data)
        
        # Merge all components
        # Assuming both datasets have 'cell_id' as the key
        if 'cell_id' not in hazard_data.columns or 'cell_id' not in exposure_data.columns:
            raise ValueError("Both datasets must have 'cell_id' column for merging")
        
        # Merge hazard data with infrastructure data
        combined_data = hazard_data.merge(
            exposure_data[['cell_id', 'exposure_score']], 
            on='cell_id', 
            how='inner'
        )
        
        combined_data = combined_data.merge(
            vulnerability_data[['cell_id', 'vulnerability_score']], 
            on='cell_id', 
            how='inner'
        )
        
        # Calculate final risk scores
        final_data = self.calculate_final_risk_scores_by_horizon(combined_data)
        
        # Add confidence scores
        final_data_with_confidence = self.add_confidence_scores(
            final_data, weather_data, infrastructure_data
        )
        
        logging.info(f"Created complete risk assessment for {len(final_data_with_confidence)} records")
        
        return final_data_with_confidence


def test_weight_sensitivity(engine: RiskScoringEngine, 
                           base_scores: Dict[str, float],
                           weight_variations: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Test sensitivity of hazard scores to weight variations.
    
    Args:
        engine: Risk scoring engine instance
        base_scores: Base stress scores for testing
        weight_variations: Dictionary of weight variations to test
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    results = []
    
    # Base case
    base_hazard = engine.calculate_hazard_score(
        base_scores['thermal'],
        base_scores['wind'],
        base_scores['precip'],
        base_scores['storm']
    )
    
    results.append({
        'scenario': 'base',
        'thermal_weight': engine.weights.thermal,
        'wind_weight': engine.weights.wind,
        'precip_weight': engine.weights.precip,
        'storm_weight': engine.weights.storm,
        'hazard_score': base_hazard,
        'score_change': 0.0
    })
    
    # Test weight variations
    for weight_name, variations in weight_variations.items():
        for variation in variations:
            # Create modified engine
            modified_config = engine.config.copy()
            modified_config['weights'][weight_name] = variation
            
            # Renormalize other weights to maintain sum = 1.0
            if weight_name in ['thermal', 'wind', 'precip', 'storm']:
                other_weights = ['thermal', 'wind', 'precip', 'storm']
                other_weights.remove(weight_name)
                
                remaining_weight = 1.0 - variation
                current_sum = sum(modified_config['weights'][w] for w in other_weights)
                
                if current_sum > 0:
                    scale_factor = remaining_weight / current_sum
                    for w in other_weights:
                        modified_config['weights'][w] *= scale_factor
            
            modified_engine = RiskScoringEngine(modified_config)
            
            # Calculate hazard score with modified weights
            modified_hazard = modified_engine.calculate_hazard_score(
                base_scores['thermal'],
                base_scores['wind'],
                base_scores['precip'],
                base_scores['storm']
            )
            
            results.append({
                'scenario': f'{weight_name}_{variation}',
                'thermal_weight': modified_engine.weights.thermal,
                'wind_weight': modified_engine.weights.wind,
                'precip_weight': modified_engine.weights.precip,
                'storm_weight': modified_engine.weights.storm,
                'hazard_score': modified_hazard,
                'score_change': modified_hazard - base_hazard
            })
    
    return pd.DataFrame(results)