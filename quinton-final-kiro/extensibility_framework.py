"""
Extensibility Framework for MISO Weather-Stress Heatmap

This module provides a plugin architecture for adding new risk components and stressors
to the MISO Weather-Stress Heatmap system. It defines standardized interfaces and
provides example implementations for extending the system.

Key Components:
- Abstract base classes for risk components
- Plugin registration and management system
- Configuration hooks for additional weights and parameters
- Example implementations for resource transition and load growth indicators
- Documentation and extension patterns

Requirements addressed: 7.1, 7.2, 7.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from enum import Enum


class ComponentType(Enum):
    """Types of risk components that can be added to the system"""
    HAZARD = "hazard"
    EXPOSURE = "exposure"
    VULNERABILITY = "vulnerability"


@runtime_checkable
class RiskComponent(Protocol):
    """
    Protocol defining the interface for risk components.
    All risk components must implement this interface.
    """
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate risk score from input data.
        
        Args:
            data: Input data for score calculation
            
        Returns:
            Risk score(s) in range [0, 1]
        """
        ...
    
    def get_component_name(self) -> str:
        """Get the name of this risk component"""
        ...
    
    def get_component_type(self) -> ComponentType:
        """Get the type of this risk component"""
        ...
    
    def get_required_columns(self) -> List[str]:
        """Get list of required data columns"""
        ...
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data meets requirements"""
        ...


@dataclass
class ComponentConfig:
    """Configuration for a risk component"""
    name: str
    component_type: ComponentType
    weight: float
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseRiskComponent(ABC):
    """
    Abstract base class for risk components.
    Provides common functionality and enforces interface compliance.
    """
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize risk component with configuration.
        
        Args:
            config: Component configuration
        """
        self.config = config
        self.name = config.name
        self.component_type = config.component_type
        self.weight = config.weight
        self.enabled = config.enabled
        self.parameters = config.parameters
        
        logging.info(f"Initialized {self.component_type.value} component: {self.name}")
    
    @abstractmethod
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """Calculate risk score from input data"""
        pass
    
    def get_component_name(self) -> str:
        """Get the name of this risk component"""
        return self.name
    
    def get_component_type(self) -> ComponentType:
        """Get the type of this risk component"""
        return self.component_type
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get list of required data columns"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data meets requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"Component {self.name} missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _validate_score_range(self, score: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Validate and clip score to [0, 1] range.
        
        Args:
            score: Raw score value(s)
            
        Returns:
            Clipped score in [0, 1] range
        """
        if isinstance(score, (np.ndarray, pd.Series)):
            clipped = np.clip(score, 0.0, 1.0)
            if not np.allclose(score, clipped, atol=1e-6):
                logging.warning(f"Component {self.name} scores clipped to [0,1] range")
            return clipped
        else:
            clipped = max(0.0, min(1.0, score))
            if abs(score - clipped) > 1e-6:
                logging.warning(f"Component {self.name} score clipped to [0,1] range")
            return clipped


class ResourceTransitionComponent(BaseRiskComponent):
    """
    Example vulnerability component for resource transition indicators.
    
    This component assesses vulnerability based on the pace of renewable energy
    transition and grid modernization in each area. Higher transition rates
    may indicate temporary vulnerability during infrastructure changes.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize resource transition component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'transition_rate_threshold': 0.1,  # 10% annual change threshold
            'modernization_lag_penalty': 0.3,  # Penalty for lagging modernization
            'baseline_transition_rate': 0.05   # 5% baseline transition rate
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate resource transition vulnerability score.
        
        Score increases with:
        - Rapid renewable capacity additions (grid stability concerns)
        - Retirement of conventional generation without replacement
        - Lack of grid modernization investments
        
        Args:
            data: DataFrame with transition indicators
            
        Returns:
            Transition vulnerability score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([0.5] * len(data), index=data.index)
            else:
                return 0.5
        
        # Calculate transition rate score
        transition_rate = data.get('renewable_transition_rate', self.parameters['baseline_transition_rate'])
        
        # Higher transition rates increase vulnerability (temporary instability)
        transition_score = np.minimum(
            transition_rate / self.parameters['transition_rate_threshold'],
            1.0
        )
        
        # Add modernization lag penalty
        modernization_score = data.get('grid_modernization_score', 0.5)  # 0.5 = neutral
        modernization_penalty = (1.0 - modernization_score) * self.parameters['modernization_lag_penalty']
        
        # Combine scores
        total_score = transition_score + modernization_penalty
        
        return self._validate_score_range(total_score)
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['renewable_transition_rate']  # Minimum required
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return ['grid_modernization_score', 'conventional_retirement_rate']


class LoadGrowthComponent(BaseRiskComponent):
    """
    Example exposure component for load growth indicators.
    
    This component assesses exposure based on projected load growth patterns,
    economic development, and population changes. Areas with rapid load growth
    may have higher exposure to supply-demand imbalances.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize load growth component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'high_growth_threshold': 0.05,  # 5% annual load growth
            'economic_growth_weight': 0.4,
            'population_growth_weight': 0.3,
            'industrial_growth_weight': 0.3
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate load growth exposure score.
        
        Score increases with:
        - Projected load growth rate
        - Economic development indicators
        - Population growth trends
        - Industrial expansion
        
        Args:
            data: DataFrame with load growth indicators
            
        Returns:
            Load growth exposure score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([0.3] * len(data), index=data.index)
            else:
                return 0.3
        
        # Base load growth score
        load_growth_rate = data['projected_load_growth_rate']
        base_score = np.minimum(
            load_growth_rate / self.parameters['high_growth_threshold'],
            1.0
        )
        
        # Economic growth factor
        economic_growth = data.get('economic_growth_rate', 0.0)
        economic_factor = np.minimum(economic_growth / 0.03, 1.0)  # 3% threshold
        
        # Population growth factor
        population_growth = data.get('population_growth_rate', 0.0)
        population_factor = np.minimum(population_growth / 0.02, 1.0)  # 2% threshold
        
        # Industrial growth factor
        industrial_growth = data.get('industrial_growth_rate', 0.0)
        industrial_factor = np.minimum(industrial_growth / 0.05, 1.0)  # 5% threshold
        
        # Weighted combination
        composite_score = (
            base_score * 0.4 +  # Base load growth gets 40% weight
            economic_factor * self.parameters['economic_growth_weight'] +
            population_factor * self.parameters['population_growth_weight'] +
            industrial_factor * self.parameters['industrial_growth_weight']
        )
        
        return self._validate_score_range(composite_score)
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['projected_load_growth_rate']
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return ['economic_growth_rate', 'population_growth_rate', 'industrial_growth_rate']


class ExtensibilityManager:
    """
    Manager class for registering and using extensible risk components.
    Provides plugin architecture for adding new risk factors to the system.
    """
    
    def __init__(self):
        """Initialize extensibility manager"""
        self.components: Dict[str, BaseRiskComponent] = {}
        self.component_configs: Dict[str, ComponentConfig] = {}
        
        logging.info("Extensibility manager initialized")
    
    def register_component(self, component: BaseRiskComponent) -> None:
        """
        Register a new risk component.
        
        Args:
            component: Risk component to register
        """
        name = component.get_component_name()
        
        if name in self.components:
            logging.warning(f"Overwriting existing component: {name}")
        
        self.components[name] = component
        self.component_configs[name] = component.config
        
        logging.info(f"Registered {component.get_component_type().value} component: {name}")
    
    def unregister_component(self, name: str) -> None:
        """
        Unregister a risk component.
        
        Args:
            name: Name of component to unregister
        """
        if name in self.components:
            component_type = self.components[name].get_component_type().value
            del self.components[name]
            del self.component_configs[name]
            logging.info(f"Unregistered {component_type} component: {name}")
        else:
            logging.warning(f"Component not found for unregistration: {name}")
    
    def get_component(self, name: str) -> Optional[BaseRiskComponent]:
        """
        Get a registered component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def list_components(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """
        List registered components, optionally filtered by type.
        
        Args:
            component_type: Optional component type filter
            
        Returns:
            List of component names
        """
        if component_type is None:
            return list(self.components.keys())
        else:
            return [
                name for name, component in self.components.items()
                if component.get_component_type() == component_type
            ]
    
    def calculate_component_scores(self, 
                                 data: pd.DataFrame,
                                 component_type: Optional[ComponentType] = None) -> Dict[str, Union[float, np.ndarray, pd.Series]]:
        """
        Calculate scores for all registered components of a given type.
        
        Args:
            data: Input data for calculations
            component_type: Optional component type filter
            
        Returns:
            Dictionary mapping component names to scores
        """
        results = {}
        
        components_to_process = (
            self.list_components(component_type) if component_type 
            else self.list_components()
        )
        
        for name in components_to_process:
            component = self.components[name]
            
            if not component.enabled:
                logging.debug(f"Skipping disabled component: {name}")
                continue
            
            try:
                score = component.calculate_score(data)
                results[name] = score
                logging.debug(f"Calculated score for component {name}")
            except Exception as e:
                logging.error(f"Error calculating score for component {name}: {e}")
                # Provide fallback score
                if isinstance(data.index, pd.Index) and len(data) > 0:
                    results[name] = pd.Series([0.5] * len(data), index=data.index)
                else:
                    results[name] = 0.5
        
        return results
    
    def get_extended_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema including all registered components.
        
        Returns:
            Extended configuration schema
        """
        schema = {
            'extended_components': {}
        }
        
        for name, config in self.component_configs.items():
            schema['extended_components'][name] = {
                'type': config.component_type.value,
                'weight': config.weight,
                'enabled': config.enabled,
                'parameters': config.parameters
            }
        
        return schema
    
    def load_components_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load and register components from configuration.
        
        Args:
            config: Configuration dictionary with component definitions
        """
        if 'extended_components' not in config:
            logging.info("No extended components found in configuration")
            return
        
        extended_config = config['extended_components']
        
        for name, component_config in extended_config.items():
            try:
                # Create component configuration
                comp_config = ComponentConfig(
                    name=name,
                    component_type=ComponentType(component_config['type']),
                    weight=component_config.get('weight', 0.1),
                    enabled=component_config.get('enabled', True),
                    parameters=component_config.get('parameters', {})
                )
                
                # Create component instance based on type
                # This is a simplified factory - in practice, you'd have a more
                # sophisticated plugin loading mechanism
                if name == 'resource_transition':
                    component = ResourceTransitionComponent(comp_config)
                elif name == 'load_growth':
                    component = LoadGrowthComponent(comp_config)
                else:
                    logging.warning(f"Unknown component type for {name}, skipping")
                    continue
                
                self.register_component(component)
                
            except Exception as e:
                logging.error(f"Error loading component {name}: {e}")


def create_example_extended_config() -> Dict[str, Any]:
    """
    Create example configuration with extended components.
    
    Returns:
        Example configuration dictionary
    """
    return {
        'extended_components': {
            'resource_transition': {
                'type': 'vulnerability',
                'weight': 0.15,
                'enabled': True,
                'parameters': {
                    'transition_rate_threshold': 0.08,
                    'modernization_lag_penalty': 0.25,
                    'baseline_transition_rate': 0.03
                }
            },
            'load_growth': {
                'type': 'exposure',
                'weight': 0.2,
                'enabled': True,
                'parameters': {
                    'high_growth_threshold': 0.04,
                    'economic_growth_weight': 0.5,
                    'population_growth_weight': 0.3,
                    'industrial_growth_weight': 0.2
                }
            }
        }
    }


def get_extension_documentation() -> str:
    """
    Get documentation for extending the system with new components.
    
    Returns:
        Documentation string
    """
    return """
# Extending the MISO Weather-Stress Heatmap System

## Overview

The extensibility framework allows you to add new risk components to the system
without modifying core code. Components can contribute to hazard, exposure, or
vulnerability scores.

## Creating a New Component

1. **Inherit from BaseRiskComponent**:
   ```python
   class MyCustomComponent(BaseRiskComponent):
       def __init__(self, config: ComponentConfig):
           super().__init__(config)
           # Initialize your component
       
       def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
           # Implement your scoring logic
           # Must return values in [0, 1] range
           pass
       
       def get_required_columns(self) -> List[str]:
           # Return list of required data columns
           return ['my_required_column']
   ```

2. **Register Your Component**:
   ```python
   manager = ExtensibilityManager()
   config = ComponentConfig(
       name='my_component',
       component_type=ComponentType.HAZARD,  # or EXPOSURE, VULNERABILITY
       weight=0.1,
       enabled=True,
       parameters={'param1': 'value1'}
   )
   component = MyCustomComponent(config)
   manager.register_component(component)
   ```

3. **Use in Configuration**:
   ```yaml
   extended_components:
     my_component:
       type: hazard
       weight: 0.1
       enabled: true
       parameters:
         param1: value1
   ```

## Component Types

- **HAZARD**: Weather-related stress factors (thermal, wind, precipitation, etc.)
- **EXPOSURE**: Population and load exposure factors
- **VULNERABILITY**: Infrastructure and grid vulnerability factors

## Best Practices

1. Always validate input data in your component
2. Return scores in [0, 1] range
3. Handle missing data gracefully
4. Log important events and errors
5. Provide meaningful parameter defaults
6. Document your component's purpose and methodology

## Example Components

The framework includes two example components:

1. **ResourceTransitionComponent**: Assesses vulnerability from renewable transition
2. **LoadGrowthComponent**: Assesses exposure from load growth patterns

These serve as templates for creating your own components.
"""