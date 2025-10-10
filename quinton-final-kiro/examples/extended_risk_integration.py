"""
Extended Risk Integration for MISO Weather-Stress Heatmap

This module integrates the extensibility framework with the existing risk scoring
system, allowing extended components to contribute to final risk calculations.

Key Components:
- Integration with existing RiskScoringEngine
- Extended configuration management
- Weighted combination of core and extended components
- Backward compatibility with existing system

Requirements addressed: 7.1, 7.2, 7.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass

from extensibility_framework import (
    ExtensibilityManager, ComponentType, ComponentConfig,
    ResourceTransitionComponent, LoadGrowthComponent
)
from risk_scoring_engine import RiskScoringEngine, RiskWeights, RiskAssessment


@dataclass
class ExtendedRiskWeights:
    """Extended risk weights including core and extended components"""
    # Core component weights (from original system)
    core_hazard_weight: float = 0.8
    core_exposure_weight: float = 0.8
    core_vulnerability_weight: float = 0.8
    
    # Extended component weights
    extended_hazard_weight: float = 0.2
    extended_exposure_weight: float = 0.2
    extended_vulnerability_weight: float = 0.2
    
    # Final blend weights (unchanged from original)
    alpha: float = 0.5  # hazard weight
    beta: float = 0.3   # exposure weight
    gamma: float = 0.2  # vulnerability weight


class ExtendedRiskScoringEngine:
    """
    Extended risk scoring engine that combines core risk components with
    pluggable extended components through the extensibility framework.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 extensibility_manager: Optional[ExtensibilityManager] = None):
        """
        Initialize extended risk scoring engine.
        
        Args:
            config: Configuration dictionary
            extensibility_manager: Optional pre-configured extensibility manager
        """
        self.config = config or self._get_default_extended_config()
        
        # Initialize core risk scoring engine
        self.core_engine = RiskScoringEngine(config)
        
        # Initialize extensibility manager
        self.extensibility_manager = extensibility_manager or ExtensibilityManager()
        
        # Load extended components from configuration
        self._load_extended_components()
        
        # Initialize extended weights
        self.extended_weights = ExtendedRiskWeights(**self.config.get('extended_weights', {}))
        
        logging.info("Extended risk scoring engine initialized")
        logging.info(f"Core component weights: H={self.extended_weights.core_hazard_weight}, "
                    f"E={self.extended_weights.core_exposure_weight}, "
                    f"V={self.extended_weights.core_vulnerability_weight}")
        logging.info(f"Extended component weights: H={self.extended_weights.extended_hazard_weight}, "
                    f"E={self.extended_weights.extended_exposure_weight}, "
                    f"V={self.extended_weights.extended_vulnerability_weight}")
    
    def _get_default_extended_config(self) -> Dict[str, Any]:
        """Get default extended configuration"""
        base_config = {
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
            },
            'extended_weights': {
                'core_hazard_weight': 0.8,
                'core_exposure_weight': 0.8,
                'core_vulnerability_weight': 0.8,
                'extended_hazard_weight': 0.2,
                'extended_exposure_weight': 0.2,
                'extended_vulnerability_weight': 0.2,
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2
            },
            'extended_components': {}
        }
        return base_config
    
    def _load_extended_components(self) -> None:
        """Load extended components from configuration"""
        if 'extended_components' in self.config:
            self.extensibility_manager.load_components_from_config(self.config)
            
            # Log loaded components
            for component_type in ComponentType:
                components = self.extensibility_manager.list_components(component_type)
                if components:
                    logging.info(f"Loaded {component_type.value} extensions: {components}")
    
    def calculate_extended_hazard_score(self, 
                                      weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate extended hazard score combining core and extended components.
        
        Args:
            weather_data: DataFrame with weather stress scores
            
        Returns:
            DataFrame with extended hazard scores
        """
        result_df = weather_data.copy()
        
        # Calculate core hazard score using existing engine
        core_hazard_df = self.core_engine.process_hazard_scores(weather_data)
        core_hazard_score = core_hazard_df['hazard_score']
        
        # Calculate extended hazard scores
        extended_scores = self.extensibility_manager.calculate_component_scores(
            weather_data, ComponentType.HAZARD
        )
        
        if extended_scores:
            # Combine extended hazard scores (weighted average)
            extended_hazard_score = self._combine_extended_scores(
                extended_scores, weather_data.index
            )
            
            # Blend core and extended scores
            final_hazard_score = (
                self.extended_weights.core_hazard_weight * core_hazard_score +
                self.extended_weights.extended_hazard_weight * extended_hazard_score
            )
            
            logging.info(f"Combined core and extended hazard scores for {len(result_df)} records")
        else:
            # No extended components, use core score only
            final_hazard_score = core_hazard_score
            logging.info("No extended hazard components, using core scores only")
        
        result_df['extended_hazard_score'] = final_hazard_score
        return result_df
    
    def calculate_extended_exposure_score(self, 
                                        infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate extended exposure score combining core and extended components.
        
        Args:
            infrastructure_data: DataFrame with infrastructure and exposure data
            
        Returns:
            DataFrame with extended exposure scores
        """
        result_df = infrastructure_data.copy()
        
        # Calculate core exposure score using existing engine
        core_exposure_df = self.core_engine.process_exposure_scores(infrastructure_data)
        core_exposure_score = core_exposure_df['exposure_score']
        
        # Calculate extended exposure scores
        extended_scores = self.extensibility_manager.calculate_component_scores(
            infrastructure_data, ComponentType.EXPOSURE
        )
        
        if extended_scores:
            # Combine extended exposure scores (weighted average)
            extended_exposure_score = self._combine_extended_scores(
                extended_scores, infrastructure_data.index
            )
            
            # Blend core and extended scores
            final_exposure_score = (
                self.extended_weights.core_exposure_weight * core_exposure_score +
                self.extended_weights.extended_exposure_weight * extended_exposure_score
            )
            
            logging.info(f"Combined core and extended exposure scores for {len(result_df)} records")
        else:
            # No extended components, use core score only
            final_exposure_score = core_exposure_score
            logging.info("No extended exposure components, using core scores only")
        
        result_df['extended_exposure_score'] = final_exposure_score
        return result_df
    
    def calculate_extended_vulnerability_score(self, 
                                             infrastructure_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate extended vulnerability score combining core and extended components.
        
        Args:
            infrastructure_data: DataFrame with infrastructure and vulnerability data
            
        Returns:
            DataFrame with extended vulnerability scores
        """
        result_df = infrastructure_data.copy()
        
        # Calculate core vulnerability score using existing engine
        core_vulnerability_df = self.core_engine.process_vulnerability_scores(infrastructure_data)
        core_vulnerability_score = core_vulnerability_df['vulnerability_score']
        
        # Calculate extended vulnerability scores
        extended_scores = self.extensibility_manager.calculate_component_scores(
            infrastructure_data, ComponentType.VULNERABILITY
        )
        
        if extended_scores:
            # Combine extended vulnerability scores (weighted average)
            extended_vulnerability_score = self._combine_extended_scores(
                extended_scores, infrastructure_data.index
            )
            
            # Blend core and extended scores
            final_vulnerability_score = (
                self.extended_weights.core_vulnerability_weight * core_vulnerability_score +
                self.extended_weights.extended_vulnerability_weight * extended_vulnerability_score
            )
            
            logging.info(f"Combined core and extended vulnerability scores for {len(result_df)} records")
        else:
            # No extended components, use core score only
            final_vulnerability_score = core_vulnerability_score
            logging.info("No extended vulnerability components, using core scores only")
        
        result_df['extended_vulnerability_score'] = final_vulnerability_score
        return result_df
    
    def _combine_extended_scores(self, 
                               extended_scores: Dict[str, Union[float, np.ndarray, pd.Series]],
                               index: pd.Index) -> pd.Series:
        """
        Combine multiple extended component scores using their weights.
        
        Args:
            extended_scores: Dictionary of component scores
            index: DataFrame index for result
            
        Returns:
            Combined extended score
        """
        if not extended_scores:
            return pd.Series([0.0] * len(index), index=index)
        
        # Get component weights
        total_weight = 0.0
        weighted_sum = pd.Series([0.0] * len(index), index=index)
        
        for component_name, score in extended_scores.items():
            component = self.extensibility_manager.get_component(component_name)
            if component and component.enabled:
                weight = component.weight
                total_weight += weight
                
                # Convert score to Series if needed
                if isinstance(score, (float, int)):
                    score_series = pd.Series([score] * len(index), index=index)
                else:
                    score_series = pd.Series(score, index=index)
                
                weighted_sum += weight * score_series
        
        # Normalize by total weight
        if total_weight > 0:
            combined_score = weighted_sum / total_weight
        else:
            combined_score = pd.Series([0.0] * len(index), index=index)
        
        # Ensure scores are in [0, 1] range
        return np.clip(combined_score, 0.0, 1.0)
    
    def calculate_extended_final_risk(self,
                                    hazard_data: pd.DataFrame,
                                    exposure_data: pd.DataFrame,
                                    vulnerability_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final extended risk scores.
        
        Args:
            hazard_data: DataFrame with extended hazard scores
            exposure_data: DataFrame with extended exposure scores
            vulnerability_data: DataFrame with extended vulnerability scores
            
        Returns:
            DataFrame with final extended risk scores
        """
        # Ensure all dataframes have the same index
        common_index = hazard_data.index.intersection(
            exposure_data.index.intersection(vulnerability_data.index)
        )
        
        if len(common_index) == 0:
            raise ValueError("No common index found between hazard, exposure, and vulnerability data")
        
        # Align dataframes to common index
        hazard_aligned = hazard_data.loc[common_index]
        exposure_aligned = exposure_data.loc[common_index]
        vulnerability_aligned = vulnerability_data.loc[common_index]
        
        # Extract extended scores
        hazard_scores = hazard_aligned['extended_hazard_score']
        exposure_scores = exposure_aligned['extended_exposure_score']
        vulnerability_scores = vulnerability_aligned['extended_vulnerability_score']
        
        # Calculate final risk using core engine formula
        final_risk = self.core_engine.calculate_final_risk_score(
            hazard_scores, exposure_scores, vulnerability_scores
        )
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'cell_id': common_index,
            'extended_hazard_score': hazard_scores,
            'extended_exposure_score': exposure_scores,
            'extended_vulnerability_score': vulnerability_scores,
            'extended_final_risk': final_risk
        }, index=common_index)
        
        logging.info(f"Calculated extended final risk scores for {len(result_df)} records")
        return result_df
    
    def get_extended_component_contributions(self, 
                                           cell_data: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of component contributions for a specific cell.
        
        Args:
            cell_data: Data for a specific grid cell
            
        Returns:
            Dictionary with component contributions by type
        """
        contributions = {
            'hazard': {},
            'exposure': {},
            'vulnerability': {}
        }
        
        # Get core component contributions
        if all(col in cell_data for col in ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy']):
            core_hazard_contrib = self.core_engine.get_hazard_component_contributions(
                cell_data['thermal_stress'],
                cell_data['wind_stress'],
                cell_data['precip_stress'],
                cell_data['storm_proxy']
            )
            contributions['hazard'].update({f"core_{k}": v for k, v in core_hazard_contrib.items()})
        
        # Get extended component contributions
        for component_type in ComponentType:
            components = self.extensibility_manager.list_components(component_type)
            for component_name in components:
                component = self.extensibility_manager.get_component(component_name)
                if component and component.enabled:
                    try:
                        # Create single-row dataframe for component calculation
                        single_row_df = pd.DataFrame([cell_data])
                        score = component.calculate_score(single_row_df)
                        
                        if isinstance(score, (pd.Series, np.ndarray)):
                            score = score[0] if len(score) > 0 else 0.0
                        
                        contribution = component.weight * score
                        contributions[component_type.value][component_name] = contribution
                        
                    except Exception as e:
                        logging.error(f"Error calculating contribution for {component_name}: {e}")
                        contributions[component_type.value][component_name] = 0.0
        
        return contributions
    
    def validate_extended_system(self, 
                               weather_data: pd.DataFrame,
                               infrastructure_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the extended risk scoring system.
        
        Args:
            weather_data: Weather data for validation
            infrastructure_data: Infrastructure data for validation
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'core_system_valid': True,
            'extended_components_valid': True,
            'integration_valid': True,
            'details': {}
        }
        
        try:
            # Validate core system
            core_hazard_validation = self.core_engine.validate_hazard_calculation(weather_data)
            core_exposure_validation = self.core_engine.validate_exposure_calculation(infrastructure_data)
            
            validation_results['details']['core_hazard'] = core_hazard_validation
            validation_results['details']['core_exposure'] = core_exposure_validation
            
            if not all(core_hazard_validation.values()) or not all(core_exposure_validation.values()):
                validation_results['core_system_valid'] = False
            
            # Validate extended components
            extended_validation = {}
            for component_type in ComponentType:
                components = self.extensibility_manager.list_components(component_type)
                for component_name in components:
                    component = self.extensibility_manager.get_component(component_name)
                    if component and component.enabled:
                        try:
                            # Test component with sample data
                            if component_type == ComponentType.HAZARD:
                                test_data = weather_data.head(5) if len(weather_data) > 0 else pd.DataFrame()
                            else:
                                test_data = infrastructure_data.head(5) if len(infrastructure_data) > 0 else pd.DataFrame()
                            
                            if len(test_data) > 0:
                                is_valid = component.validate_data(test_data)
                                extended_validation[component_name] = is_valid
                                
                                if not is_valid:
                                    validation_results['extended_components_valid'] = False
                            
                        except Exception as e:
                            logging.error(f"Error validating component {component_name}: {e}")
                            extended_validation[component_name] = False
                            validation_results['extended_components_valid'] = False
            
            validation_results['details']['extended_components'] = extended_validation
            
            # Test integration
            try:
                if len(weather_data) > 0 and len(infrastructure_data) > 0:
                    # Test extended score calculations
                    sample_weather = weather_data.head(3)
                    sample_infra = infrastructure_data.head(3)
                    
                    extended_hazard = self.calculate_extended_hazard_score(sample_weather)
                    extended_exposure = self.calculate_extended_exposure_score(sample_infra)
                    extended_vulnerability = self.calculate_extended_vulnerability_score(sample_infra)
                    
                    # Check that extended scores are in valid range
                    hazard_valid = (extended_hazard['extended_hazard_score'] >= 0).all() and (extended_hazard['extended_hazard_score'] <= 1).all()
                    exposure_valid = (extended_exposure['extended_exposure_score'] >= 0).all() and (extended_exposure['extended_exposure_score'] <= 1).all()
                    vulnerability_valid = (extended_vulnerability['extended_vulnerability_score'] >= 0).all() and (extended_vulnerability['extended_vulnerability_score'] <= 1).all()
                    
                    validation_results['details']['integration'] = {
                        'hazard_scores_valid': hazard_valid,
                        'exposure_scores_valid': exposure_valid,
                        'vulnerability_scores_valid': vulnerability_valid
                    }
                    
                    if not (hazard_valid and exposure_valid and vulnerability_valid):
                        validation_results['integration_valid'] = False
                
            except Exception as e:
                logging.error(f"Error validating integration: {e}")
                validation_results['integration_valid'] = False
                validation_results['details']['integration_error'] = str(e)
        
        except Exception as e:
            logging.error(f"Error in extended system validation: {e}")
            validation_results['core_system_valid'] = False
            validation_results['extended_components_valid'] = False
            validation_results['integration_valid'] = False
            validation_results['details']['validation_error'] = str(e)
        
        return validation_results


def create_extended_system_example() -> ExtendedRiskScoringEngine:
    """
    Create an example extended risk scoring system with sample components.
    
    Returns:
        Configured extended risk scoring engine
    """
    # Create configuration with extended components
    config = {
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
        },
        'extended_weights': {
            'core_hazard_weight': 0.85,
            'core_exposure_weight': 0.8,
            'core_vulnerability_weight': 0.75,
            'extended_hazard_weight': 0.15,
            'extended_exposure_weight': 0.2,
            'extended_vulnerability_weight': 0.25
        },
        'extended_components': {
            'resource_transition': {
                'type': 'vulnerability',
                'weight': 0.6,
                'enabled': True,
                'parameters': {
                    'transition_rate_threshold': 0.08,
                    'modernization_lag_penalty': 0.25
                }
            },
            'load_growth': {
                'type': 'exposure',
                'weight': 0.4,
                'enabled': True,
                'parameters': {
                    'high_growth_threshold': 0.04,
                    'economic_growth_weight': 0.5
                }
            }
        }
    }
    
    # Create extended risk scoring engine
    extended_engine = ExtendedRiskScoringEngine(config)
    
    logging.info("Created example extended risk scoring system")
    return extended_engine