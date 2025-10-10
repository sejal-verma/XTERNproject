"""
Extension Examples for MISO Weather-Stress Heatmap

This module provides concrete examples of how to extend the MISO Weather-Stress
Heatmap system with new risk components. These examples serve as templates and
documentation for creating custom extensions.

Key Examples:
- Cyber security risk component
- Market volatility component  
- Seasonal demand pattern component
- Infrastructure age component

Requirements addressed: 7.1, 7.2, 7.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timedelta

from extensibility_framework import (
    BaseRiskComponent, ComponentConfig, ComponentType
)


class CyberSecurityRiskComponent(BaseRiskComponent):
    """
    Example vulnerability component for cyber security risk assessment.
    
    This component assesses vulnerability based on cyber security indicators
    such as recent incidents, security infrastructure maturity, and threat levels.
    Higher cyber risk increases grid vulnerability during stress events.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize cyber security risk component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'incident_decay_days': 90,      # Days for incident impact to decay
            'high_threat_threshold': 0.7,   # High threat level threshold
            'security_maturity_weight': 0.4,
            'incident_history_weight': 0.3,
            'threat_level_weight': 0.3
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate cyber security vulnerability score.
        
        Score increases with:
        - Recent cyber security incidents
        - Higher threat intelligence levels
        - Lower security infrastructure maturity
        - Proximity to critical infrastructure
        
        Args:
            data: DataFrame with cyber security indicators
            
        Returns:
            Cyber security vulnerability score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            baseline_score = 0.3  # Moderate baseline cyber risk
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([baseline_score] * len(data), index=data.index)
            else:
                return baseline_score
        
        # Security maturity score (inverted - lower maturity = higher risk)
        security_maturity = data.get('security_maturity_score', 0.5)
        maturity_risk = 1.0 - security_maturity
        
        # Recent incident impact (decaying over time)
        incident_score = self._calculate_incident_impact(data)
        
        # Current threat level
        threat_level = data.get('threat_level', 0.3)
        
        # Weighted combination
        cyber_risk_score = (
            self.parameters['security_maturity_weight'] * maturity_risk +
            self.parameters['incident_history_weight'] * incident_score +
            self.parameters['threat_level_weight'] * threat_level
        )
        
        return self._validate_score_range(cyber_risk_score)
    
    def _calculate_incident_impact(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """Calculate impact of recent cyber security incidents"""
        if 'days_since_last_incident' not in data.columns:
            return 0.0
        
        days_since = data['days_since_last_incident']
        incident_severity = data.get('last_incident_severity', 0.5)
        
        # Exponential decay of incident impact
        decay_factor = np.exp(-days_since / self.parameters['incident_decay_days'])
        incident_impact = incident_severity * decay_factor
        
        return incident_impact
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return []  # All columns are optional with defaults
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return [
            'security_maturity_score',
            'threat_level',
            'days_since_last_incident',
            'last_incident_severity',
            'critical_infrastructure_proximity'
        ]


class MarketVolatilityComponent(BaseRiskComponent):
    """
    Example exposure component for electricity market volatility.
    
    This component assesses exposure based on market conditions that could
    amplify the impact of grid stress events, such as price volatility,
    supply-demand imbalances, and financial transmission rights.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize market volatility component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'high_volatility_threshold': 0.3,  # 30% price volatility threshold
            'price_spike_threshold': 2.0,      # 2x average price = spike
            'congestion_weight': 0.4,
            'volatility_weight': 0.3,
            'liquidity_weight': 0.3
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate market volatility exposure score.
        
        Score increases with:
        - Higher electricity price volatility
        - Transmission congestion frequency
        - Lower market liquidity
        - Recent price spikes
        
        Args:
            data: DataFrame with market indicators
            
        Returns:
            Market volatility exposure score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            baseline_score = 0.2  # Low baseline market exposure
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([baseline_score] * len(data), index=data.index)
            else:
                return baseline_score
        
        # Price volatility score
        price_volatility = data.get('price_volatility_30d', 0.1)
        volatility_score = np.minimum(
            price_volatility / self.parameters['high_volatility_threshold'],
            1.0
        )
        
        # Transmission congestion score
        congestion_frequency = data.get('congestion_frequency', 0.1)
        congestion_score = np.minimum(congestion_frequency / 0.2, 1.0)  # 20% threshold
        
        # Market liquidity score (inverted - lower liquidity = higher exposure)
        market_liquidity = data.get('market_liquidity_index', 0.7)
        liquidity_risk = 1.0 - market_liquidity
        
        # Weighted combination
        market_exposure_score = (
            self.parameters['volatility_weight'] * volatility_score +
            self.parameters['congestion_weight'] * congestion_score +
            self.parameters['liquidity_weight'] * liquidity_risk
        )
        
        return self._validate_score_range(market_exposure_score)
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return []  # All columns are optional with defaults
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return [
            'price_volatility_30d',
            'congestion_frequency',
            'market_liquidity_index',
            'recent_price_spikes',
            'ftr_coverage_ratio'
        ]


class SeasonalDemandComponent(BaseRiskComponent):
    """
    Example hazard component for seasonal demand patterns.
    
    This component assesses hazard based on seasonal demand stress patterns
    that could coincide with weather events, such as peak cooling/heating
    demand periods and seasonal industrial activity.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize seasonal demand component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'peak_demand_threshold': 0.9,    # 90% of peak capacity
            'seasonal_factor_weight': 0.5,
            'demand_growth_weight': 0.3,
            'industrial_cycle_weight': 0.2
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate seasonal demand hazard score.
        
        Score increases with:
        - Proximity to seasonal peak demand periods
        - Higher demand growth trends
        - Industrial activity cycles
        - Cooling/heating degree day accumulation
        
        Args:
            data: DataFrame with seasonal demand indicators
            
        Returns:
            Seasonal demand hazard score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            baseline_score = 0.4  # Moderate baseline seasonal risk
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([baseline_score] * len(data), index=data.index)
            else:
                return baseline_score
        
        # Seasonal demand factor (based on time of year)
        seasonal_factor = self._calculate_seasonal_factor(data)
        
        # Demand growth trend
        demand_growth = data.get('demand_growth_rate', 0.02)  # 2% baseline
        growth_score = np.minimum(demand_growth / 0.05, 1.0)  # 5% threshold
        
        # Industrial cycle factor
        industrial_activity = data.get('industrial_activity_index', 0.5)
        
        # Weighted combination
        seasonal_hazard_score = (
            self.parameters['seasonal_factor_weight'] * seasonal_factor +
            self.parameters['demand_growth_weight'] * growth_score +
            self.parameters['industrial_cycle_weight'] * industrial_activity
        )
        
        return self._validate_score_range(seasonal_hazard_score)
    
    def _calculate_seasonal_factor(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """Calculate seasonal demand stress factor"""
        if 'day_of_year' in data.columns:
            day_of_year = data['day_of_year']
        else:
            # Use current date if not provided
            current_day = datetime.now().timetuple().tm_yday
            if isinstance(data.index, pd.Index) and len(data) > 0:
                day_of_year = pd.Series([current_day] * len(data), index=data.index)
            else:
                day_of_year = current_day
        
        # Summer peak (day 180-240, roughly June-August)
        summer_peak = np.where(
            (day_of_year >= 180) & (day_of_year <= 240),
            1.0 - np.abs(day_of_year - 210) / 30,  # Peak around July 30
            0.0
        )
        
        # Winter peak (day 1-60 and 330-365, roughly Dec-Feb)
        winter_peak = np.where(
            (day_of_year <= 60) | (day_of_year >= 330),
            1.0 - np.minimum(np.abs(day_of_year - 15), np.abs(day_of_year - 345)) / 45,
            0.0
        )
        
        # Return maximum of summer and winter peaks
        if isinstance(day_of_year, (np.ndarray, pd.Series)):
            return np.maximum(summer_peak, winter_peak)
        else:
            return max(summer_peak, winter_peak)
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return []  # All columns are optional with defaults
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return [
            'day_of_year',
            'demand_growth_rate',
            'industrial_activity_index',
            'cooling_degree_days',
            'heating_degree_days'
        ]


class InfrastructureAgeComponent(BaseRiskComponent):
    """
    Example vulnerability component for infrastructure age and condition.
    
    This component assesses vulnerability based on the age and condition of
    grid infrastructure, which affects reliability during stress events.
    Older infrastructure typically has higher failure rates.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize infrastructure age component"""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'old_infrastructure_threshold': 30,  # 30 years
            'critical_age_threshold': 50,        # 50 years = critical
            'condition_weight': 0.5,
            'age_weight': 0.3,
            'maintenance_weight': 0.2
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate infrastructure age vulnerability score.
        
        Score increases with:
        - Higher average infrastructure age
        - Poorer infrastructure condition ratings
        - Deferred maintenance backlogs
        - Higher failure rates
        
        Args:
            data: DataFrame with infrastructure age indicators
            
        Returns:
            Infrastructure age vulnerability score [0, 1]
        """
        if not self.validate_data(data):
            # Return baseline score if data is invalid
            baseline_score = 0.4  # Moderate baseline infrastructure risk
            if isinstance(data.index, pd.Index) and len(data) > 0:
                return pd.Series([baseline_score] * len(data), index=data.index)
            else:
                return baseline_score
        
        # Infrastructure age score
        avg_age = data.get('avg_infrastructure_age', 25)  # 25 year baseline
        age_score = np.minimum(
            avg_age / self.parameters['critical_age_threshold'],
            1.0
        )
        
        # Infrastructure condition score (inverted - poor condition = high risk)
        condition_rating = data.get('infrastructure_condition_rating', 0.6)
        condition_risk = 1.0 - condition_rating
        
        # Maintenance backlog score
        maintenance_backlog = data.get('maintenance_backlog_ratio', 0.1)
        maintenance_risk = np.minimum(maintenance_backlog / 0.3, 1.0)  # 30% threshold
        
        # Weighted combination
        infrastructure_vulnerability_score = (
            self.parameters['age_weight'] * age_score +
            self.parameters['condition_weight'] * condition_risk +
            self.parameters['maintenance_weight'] * maintenance_risk
        )
        
        return self._validate_score_range(infrastructure_vulnerability_score)
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return []  # All columns are optional with defaults
    
    def get_optional_columns(self) -> List[str]:
        """Get optional data columns that enhance scoring"""
        return [
            'avg_infrastructure_age',
            'infrastructure_condition_rating',
            'maintenance_backlog_ratio',
            'failure_rate_trend',
            'replacement_schedule_adherence'
        ]


def create_comprehensive_extension_example() -> Dict[str, Any]:
    """
    Create a comprehensive example configuration with multiple extension types.
    
    Returns:
        Configuration dictionary with all example extensions
    """
    return {
        'extended_components': {
            'cyber_security_risk': {
                'type': 'vulnerability',
                'weight': 0.15,
                'enabled': True,
                'parameters': {
                    'incident_decay_days': 60,
                    'high_threat_threshold': 0.8,
                    'security_maturity_weight': 0.5,
                    'incident_history_weight': 0.3,
                    'threat_level_weight': 0.2
                }
            },
            'market_volatility': {
                'type': 'exposure',
                'weight': 0.25,
                'enabled': True,
                'parameters': {
                    'high_volatility_threshold': 0.25,
                    'price_spike_threshold': 1.8,
                    'congestion_weight': 0.5,
                    'volatility_weight': 0.3,
                    'liquidity_weight': 0.2
                }
            },
            'seasonal_demand': {
                'type': 'hazard',
                'weight': 0.1,
                'enabled': True,
                'parameters': {
                    'peak_demand_threshold': 0.85,
                    'seasonal_factor_weight': 0.6,
                    'demand_growth_weight': 0.25,
                    'industrial_cycle_weight': 0.15
                }
            },
            'infrastructure_age': {
                'type': 'vulnerability',
                'weight': 0.2,
                'enabled': True,
                'parameters': {
                    'old_infrastructure_threshold': 25,
                    'critical_age_threshold': 45,
                    'condition_weight': 0.6,
                    'age_weight': 0.25,
                    'maintenance_weight': 0.15
                }
            }
        }
    }


def get_extension_implementation_guide() -> str:
    """
    Get detailed implementation guide for creating extensions.
    
    Returns:
        Implementation guide string
    """
    return """
# Extension Implementation Guide

## Step-by-Step Component Creation

### 1. Define Your Component Class

```python
class MyRiskComponent(BaseRiskComponent):
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        
        # Define default parameters
        default_params = {
            'threshold_value': 0.5,
            'weight_factor': 0.3
        }
        
        # Merge with provided parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
```

### 2. Implement Required Methods

```python
def calculate_score(self, data: pd.DataFrame) -> Union[float, np.ndarray, pd.Series]:
    # Validate input data
    if not self.validate_data(data):
        return self._get_baseline_score(data)
    
    # Extract relevant data columns
    input_value = data.get('my_input_column', default_value)
    
    # Apply your scoring logic
    score = self._apply_scoring_logic(input_value)
    
    # Validate and return score in [0, 1] range
    return self._validate_score_range(score)

def get_required_columns(self) -> List[str]:
    return ['my_required_column']  # List required columns
```

### 3. Register Your Component

```python
# Create component configuration
config = ComponentConfig(
    name='my_component',
    component_type=ComponentType.VULNERABILITY,  # or HAZARD, EXPOSURE
    weight=0.2,
    enabled=True,
    parameters={'threshold_value': 0.7}
)

# Create and register component
component = MyRiskComponent(config)
extensibility_manager.register_component(component)
```

### 4. Add to Configuration

```yaml
extended_components:
  my_component:
    type: vulnerability
    weight: 0.2
    enabled: true
    parameters:
      threshold_value: 0.7
      weight_factor: 0.4
```

## Best Practices

1. **Data Validation**: Always validate input data and provide fallback values
2. **Score Range**: Ensure scores are always in [0, 1] range
3. **Error Handling**: Handle missing data and calculation errors gracefully
4. **Logging**: Log important events and parameter values
5. **Documentation**: Document your component's purpose and methodology
6. **Testing**: Test with various data scenarios and edge cases

## Common Patterns

### Linear Scaling
```python
def linear_scale(value, min_val, max_val):
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
```

### Exponential Decay
```python
def exponential_decay(value, decay_rate):
    return np.exp(-value / decay_rate)
```

### Threshold-Based Scoring
```python
def threshold_score(value, threshold, max_score=1.0):
    return np.minimum(value / threshold, max_score)
```

### Weighted Combination
```python
def weighted_combination(values, weights):
    return np.sum([w * v for w, v in zip(weights, values)])
```
"""


def create_sample_extension_data() -> pd.DataFrame:
    """
    Create sample data for testing extension components.
    
    Returns:
        DataFrame with sample extension data
    """
    np.random.seed(42)
    n_cells = 100
    
    sample_data = pd.DataFrame({
        'cell_id': [f'cell_{i:03d}' for i in range(n_cells)],
        
        # Cyber security indicators
        'security_maturity_score': np.random.uniform(0.3, 0.9, n_cells),
        'threat_level': np.random.uniform(0.1, 0.8, n_cells),
        'days_since_last_incident': np.random.exponential(120, n_cells),
        'last_incident_severity': np.random.uniform(0.2, 0.9, n_cells),
        
        # Market volatility indicators
        'price_volatility_30d': np.random.uniform(0.05, 0.5, n_cells),
        'congestion_frequency': np.random.uniform(0.0, 0.3, n_cells),
        'market_liquidity_index': np.random.uniform(0.4, 0.95, n_cells),
        
        # Seasonal demand indicators
        'day_of_year': np.random.randint(1, 366, n_cells),
        'demand_growth_rate': np.random.uniform(-0.01, 0.08, n_cells),
        'industrial_activity_index': np.random.uniform(0.2, 0.8, n_cells),
        
        # Infrastructure age indicators
        'avg_infrastructure_age': np.random.uniform(10, 60, n_cells),
        'infrastructure_condition_rating': np.random.uniform(0.3, 0.9, n_cells),
        'maintenance_backlog_ratio': np.random.uniform(0.0, 0.4, n_cells),
        
        # Resource transition indicators (from main framework)
        'renewable_transition_rate': np.random.uniform(0.0, 0.15, n_cells),
        'grid_modernization_score': np.random.uniform(0.2, 0.8, n_cells),
        
        # Load growth indicators (from main framework)
        'projected_load_growth_rate': np.random.uniform(-0.02, 0.1, n_cells),
        'economic_growth_rate': np.random.uniform(-0.01, 0.06, n_cells),
        'population_growth_rate': np.random.uniform(-0.005, 0.03, n_cells)
    })
    
    sample_data.set_index('cell_id', inplace=True)
    return sample_data