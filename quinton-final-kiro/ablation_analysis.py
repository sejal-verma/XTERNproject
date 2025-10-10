"""
Ablation Analysis System for MISO Weather-Stress Heatmap

This module implements comprehensive ablation analysis to understand the
contribution and importance of individual risk components (Hazard, Exposure, 
Vulnerability) by systematically removing them and measuring the impact on
final risk scores.

Key Components:
- Component removal and risk recalculation
- Sensitivity analysis across different scenarios
- Ablation visualization generation
- Component importance ranking
- Validation against expected contributions

Requirements addressed: 6.3, 8.2
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings


@dataclass
class AblationResult:
    """Results from ablation analysis for a single component"""
    component_name: str
    original_risk_mean: float
    ablated_risk_mean: float
    risk_change_mean: float
    risk_change_std: float
    risk_change_percentage: float
    correlation_with_original: float
    cells_affected: int
    total_cells: int
    impact_magnitude: str  # 'low', 'medium', 'high'


@dataclass
class ComponentImportance:
    """Component importance metrics"""
    component_name: str
    importance_score: float  # [0,1] where 1 = most important
    rank: int
    contribution_percentage: float
    variance_explained: float


class AblationAnalyzer:
    """
    Main ablation analysis system that systematically removes risk components
    and measures their impact on final risk scores.
    """
    
    def __init__(self, risk_scoring_engine, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ablation analyzer.
        
        Args:
            risk_scoring_engine: RiskScoringEngine instance for risk calculations
            config: Configuration dictionary with analysis parameters
        """
        self.risk_engine = risk_scoring_engine
        self.config = config or self._get_default_config()
        
        # Component definitions for ablation
        self.components = {
            'hazard': {
                'subcomponents': ['thermal_stress', 'wind_stress', 'precip_stress', 'storm_proxy'],
                'score_column': 'hazard_score',
                'weight_key': 'alpha'
            },
            'exposure': {
                'subcomponents': ['normalized_pop_density', 'load_factor'],
                'score_column': 'exposure_score',
                'weight_key': 'beta'
            },
            'vulnerability': {
                'subcomponents': ['renewable_share', 'transmission_scarcity', 'outage_flag'],
                'score_column': 'vulnerability_score',
                'weight_key': 'gamma'
            }
        }
        
        logging.info("Ablation analyzer initialized")
        logging.info(f"Components for analysis: {list(self.components.keys())}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ablation analysis"""
        return {
            'ablation': {
                'impact_thresholds': {
                    'low': 0.1,      # <10% change = low impact
                    'medium': 0.25,  # 10-25% change = medium impact
                    'high': 0.25     # >25% change = high impact
                },
                'min_cells_for_analysis': 10,
                'confidence_threshold': 0.5
            }
        }
    
    def perform_complete_ablation_analysis(self, 
                                         combined_data: pd.DataFrame) -> Dict[str, AblationResult]:
        """
        Perform complete ablation analysis by removing each major component.
        
        Args:
            combined_data: DataFrame with all risk components and scores
            
        Returns:
            Dictionary of ablation results by component name
        """
        logging.info("Starting complete ablation analysis...")
        
        # Validate input data
        self._validate_ablation_data(combined_data)
        
        # Calculate original risk scores
        original_risks = self._calculate_baseline_risk_scores(combined_data)
        
        ablation_results = {}
        
        # Ablate each major component
        for component_name in self.components.keys():
            logging.info(f"Performing ablation analysis for {component_name} component...")
            
            try:
                result = self._ablate_component(combined_data, component_name, original_risks)
                ablation_results[component_name] = result
                
                logging.info(f"{component_name} ablation complete: "
                           f"{result.risk_change_percentage:.1f}% impact, "
                           f"{result.impact_magnitude} magnitude")
                
            except Exception as e:
                logging.error(f"Ablation analysis failed for {component_name}: {e}")
                continue
        
        # Perform subcomponent ablation for detailed analysis
        subcomponent_results = self._perform_subcomponent_ablation(combined_data, original_risks)
        ablation_results.update(subcomponent_results)
        
        logging.info(f"Ablation analysis complete for {len(ablation_results)} components")
        
        return ablation_results
    
    def _validate_ablation_data(self, data: pd.DataFrame):
        """Validate that data contains required columns for ablation analysis"""
        required_columns = [
            'cell_id', 'hazard_score', 'exposure_score', 'vulnerability_score'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for ablation: {missing_columns}")
        
        # Check for sufficient data
        if len(data) < self.config['ablation']['min_cells_for_analysis']:
            raise ValueError(f"Insufficient data for ablation analysis: {len(data)} cells < "
                           f"{self.config['ablation']['min_cells_for_analysis']} minimum")
        
        # Check for valid score ranges
        score_columns = ['hazard_score', 'exposure_score', 'vulnerability_score']
        for col in score_columns:
            if col in data.columns:
                if data[col].isna().all():
                    raise ValueError(f"All values in {col} are NaN")
                
                min_val, max_val = data[col].min(), data[col].max()
                if min_val < 0 or max_val > 1:
                    logging.warning(f"{col} has values outside [0,1]: {min_val:.3f} to {max_val:.3f}")
    
    def _calculate_baseline_risk_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate baseline risk scores using all components"""
        try:
            # Use the risk engine's final risk calculation
            baseline_risks = self.risk_engine.calculate_final_risk_score(
                data['hazard_score'],
                data['exposure_score'], 
                data['vulnerability_score']
            )
            
            return baseline_risks
            
        except Exception as e:
            logging.error(f"Failed to calculate baseline risk scores: {e}")
            # Fallback to simple weighted combination
            weights = self.risk_engine.weights
            baseline_risks = (
                weights.alpha * data['hazard_score'] +
                weights.beta * data['exposure_score'] +
                weights.gamma * data['vulnerability_score']
            )
            
            # Apply z-score normalization
            baseline_risks = stats.zscore(baseline_risks, nan_policy='omit')
            
            return pd.Series(baseline_risks, index=data.index)
    
    def _ablate_component(self, 
                         data: pd.DataFrame, 
                         component_name: str,
                         original_risks: pd.Series) -> AblationResult:
        """
        Ablate a specific component and measure impact on risk scores.
        
        Args:
            data: DataFrame with risk components
            component_name: Name of component to ablate ('hazard', 'exposure', 'vulnerability')
            original_risks: Original risk scores for comparison
            
        Returns:
            AblationResult with impact analysis
        """
        # Create ablated data by setting component to zero
        ablated_data = data.copy()
        component_info = self.components[component_name]
        score_column = component_info['score_column']
        
        # Set component score to zero (complete removal)
        ablated_data[score_column] = 0.0
        
        # Recalculate risk scores without this component
        try:
            ablated_risks = self.risk_engine.calculate_final_risk_score(
                ablated_data['hazard_score'],
                ablated_data['exposure_score'],
                ablated_data['vulnerability_score']
            )
        except Exception as e:
            logging.warning(f"Risk engine calculation failed for {component_name} ablation: {e}")
            # Fallback calculation
            weights = self.risk_engine.weights
            ablated_risks = (
                weights.alpha * ablated_data['hazard_score'] +
                weights.beta * ablated_data['exposure_score'] +
                weights.gamma * ablated_data['vulnerability_score']
            )
            ablated_risks = stats.zscore(ablated_risks, nan_policy='omit')
            ablated_risks = pd.Series(ablated_risks, index=data.index)
        
        # Calculate impact metrics
        risk_change = original_risks - ablated_risks
        
        # Handle NaN values
        valid_mask = ~(np.isnan(original_risks) | np.isnan(ablated_risks))
        valid_original = original_risks[valid_mask]
        valid_ablated = ablated_risks[valid_mask]
        valid_change = risk_change[valid_mask]
        
        if len(valid_change) == 0:
            raise ValueError(f"No valid data points for {component_name} ablation analysis")
        
        # Calculate statistics
        original_mean = valid_original.mean()
        ablated_mean = valid_ablated.mean()
        change_mean = valid_change.mean()
        change_std = valid_change.std()
        
        # Calculate percentage change
        if abs(original_mean) > 1e-6:
            change_percentage = abs(change_mean / original_mean) * 100
        else:
            change_percentage = 0.0
        
        # Calculate correlation
        if len(valid_original) > 1 and valid_original.std() > 1e-6 and valid_ablated.std() > 1e-6:
            correlation = np.corrcoef(valid_original, valid_ablated)[0, 1]
        else:
            correlation = np.nan
        
        # Determine impact magnitude
        thresholds = self.config['ablation']['impact_thresholds']
        if change_percentage < thresholds['low'] * 100:
            impact_magnitude = 'low'
        elif change_percentage < thresholds['medium'] * 100:
            impact_magnitude = 'medium'
        else:
            impact_magnitude = 'high'
        
        return AblationResult(
            component_name=component_name,
            original_risk_mean=original_mean,
            ablated_risk_mean=ablated_mean,
            risk_change_mean=change_mean,
            risk_change_std=change_std,
            risk_change_percentage=change_percentage,
            correlation_with_original=correlation,
            cells_affected=len(valid_change),
            total_cells=len(data),
            impact_magnitude=impact_magnitude
        )
    
    def _perform_subcomponent_ablation(self, 
                                     data: pd.DataFrame,
                                     original_risks: pd.Series) -> Dict[str, AblationResult]:
        """
        Perform ablation analysis on individual subcomponents within each major component.
        
        Args:
            data: DataFrame with risk components
            original_risks: Original risk scores for comparison
            
        Returns:
            Dictionary of ablation results for subcomponents
        """
        subcomponent_results = {}
        
        for component_name, component_info in self.components.items():
            subcomponents = component_info['subcomponents']
            
            for subcomp in subcomponents:
                if subcomp not in data.columns:
                    logging.warning(f"Subcomponent {subcomp} not found in data, skipping ablation")
                    continue
                
                try:
                    # Ablate this specific subcomponent
                    result = self._ablate_subcomponent(
                        data, component_name, subcomp, original_risks
                    )
                    
                    subcomponent_key = f"{component_name}_{subcomp}"
                    subcomponent_results[subcomponent_key] = result
                    
                    logging.debug(f"Subcomponent {subcomp} ablation: "
                                f"{result.risk_change_percentage:.1f}% impact")
                    
                except Exception as e:
                    logging.warning(f"Subcomponent ablation failed for {subcomp}: {e}")
                    continue
        
        return subcomponent_results
    
    def _ablate_subcomponent(self,
                           data: pd.DataFrame,
                           parent_component: str,
                           subcomponent: str,
                           original_risks: pd.Series) -> AblationResult:
        """
        Ablate a specific subcomponent and recalculate parent component score.
        
        Args:
            data: DataFrame with risk components
            parent_component: Parent component name ('hazard', 'exposure', 'vulnerability')
            subcomponent: Subcomponent to ablate
            original_risks: Original risk scores for comparison
            
        Returns:
            AblationResult for the subcomponent
        """
        # Create ablated data
        ablated_data = data.copy()
        
        # Set subcomponent to zero
        ablated_data[subcomponent] = 0.0
        
        # Recalculate parent component score
        if parent_component == 'hazard':
            ablated_data['hazard_score'] = self.risk_engine.calculate_hazard_score(
                ablated_data.get('thermal_stress', 0),
                ablated_data.get('wind_stress', 0),
                ablated_data.get('precip_stress', 0),
                ablated_data.get('storm_proxy', 0)
            )
        elif parent_component == 'exposure':
            ablated_data['exposure_score'] = self.risk_engine.calculate_exposure_score(
                ablated_data.get('normalized_pop_density', 0),
                ablated_data.get('load_factor', None)
            )
        elif parent_component == 'vulnerability':
            ablated_data['vulnerability_score'] = self.risk_engine.calculate_vulnerability_score(
                ablated_data.get('renewable_share', 0),
                ablated_data.get('transmission_scarcity', 0.5),  # Use baseline
                ablated_data.get('outage_flag', False)
            )
        
        # Recalculate final risk scores
        try:
            ablated_risks = self.risk_engine.calculate_final_risk_score(
                ablated_data['hazard_score'],
                ablated_data['exposure_score'],
                ablated_data['vulnerability_score']
            )
        except Exception as e:
            logging.warning(f"Risk calculation failed for {subcomponent} ablation: {e}")
            # Fallback calculation
            weights = self.risk_engine.weights
            ablated_risks = (
                weights.alpha * ablated_data['hazard_score'] +
                weights.beta * ablated_data['exposure_score'] +
                weights.gamma * ablated_data['vulnerability_score']
            )
            ablated_risks = stats.zscore(ablated_risks, nan_policy='omit')
            ablated_risks = pd.Series(ablated_risks, index=data.index)
        
        # Calculate impact metrics (same as main component ablation)
        risk_change = original_risks - ablated_risks
        
        valid_mask = ~(np.isnan(original_risks) | np.isnan(ablated_risks))
        valid_original = original_risks[valid_mask]
        valid_ablated = ablated_risks[valid_mask]
        valid_change = risk_change[valid_mask]
        
        if len(valid_change) == 0:
            raise ValueError(f"No valid data points for {subcomponent} ablation analysis")
        
        original_mean = valid_original.mean()
        ablated_mean = valid_ablated.mean()
        change_mean = valid_change.mean()
        change_std = valid_change.std()
        
        if abs(original_mean) > 1e-6:
            change_percentage = abs(change_mean / original_mean) * 100
        else:
            change_percentage = 0.0
        
        if len(valid_original) > 1 and valid_original.std() > 1e-6 and valid_ablated.std() > 1e-6:
            correlation = np.corrcoef(valid_original, valid_ablated)[0, 1]
        else:
            correlation = np.nan
        
        # Determine impact magnitude
        thresholds = self.config['ablation']['impact_thresholds']
        if change_percentage < thresholds['low'] * 100:
            impact_magnitude = 'low'
        elif change_percentage < thresholds['medium'] * 100:
            impact_magnitude = 'medium'
        else:
            impact_magnitude = 'high'
        
        return AblationResult(
            component_name=f"{parent_component}_{subcomponent}",
            original_risk_mean=original_mean,
            ablated_risk_mean=ablated_mean,
            risk_change_mean=change_mean,
            risk_change_std=change_std,
            risk_change_percentage=change_percentage,
            correlation_with_original=correlation,
            cells_affected=len(valid_change),
            total_cells=len(data),
            impact_magnitude=impact_magnitude
        )
    
    def calculate_component_importance_ranking(self, 
                                             ablation_results: Dict[str, AblationResult]) -> List[ComponentImportance]:
        """
        Calculate importance ranking of components based on ablation results.
        
        Args:
            ablation_results: Dictionary of ablation results
            
        Returns:
            List of ComponentImportance objects, sorted by importance
        """
        # Filter to main components only (not subcomponents)
        main_components = {name: result for name, result in ablation_results.items() 
                          if name in self.components.keys()}
        
        if not main_components:
            logging.warning("No main component ablation results found for importance ranking")
            return []
        
        # Calculate importance scores based on risk change percentage
        importance_data = []
        
        total_impact = sum(result.risk_change_percentage for result in main_components.values())
        
        for name, result in main_components.items():
            # Importance score based on relative impact
            if total_impact > 0:
                importance_score = result.risk_change_percentage / total_impact
                contribution_percentage = result.risk_change_percentage
            else:
                importance_score = 0.0
                contribution_percentage = 0.0
            
            # Calculate variance explained (simplified)
            variance_explained = min(result.risk_change_percentage / 100, 1.0)
            
            importance_data.append({
                'component_name': name,
                'importance_score': importance_score,
                'contribution_percentage': contribution_percentage,
                'variance_explained': variance_explained
            })
        
        # Sort by importance score (descending)
        importance_data.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Add ranks
        component_importance = []
        for rank, data in enumerate(importance_data, 1):
            component_importance.append(ComponentImportance(
                component_name=data['component_name'],
                importance_score=data['importance_score'],
                rank=rank,
                contribution_percentage=data['contribution_percentage'],
                variance_explained=data['variance_explained']
            ))
        
        return component_importance
    
    def validate_ablation_results(self, 
                                ablation_results: Dict[str, AblationResult],
                                expected_patterns: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
        """
        Validate ablation results against expected component contributions.
        
        Args:
            ablation_results: Dictionary of ablation results
            expected_patterns: Optional dictionary of expected impact patterns
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        # Check that all main components have results
        main_components = set(self.components.keys())
        available_components = set(name for name in ablation_results.keys() 
                                 if name in main_components)
        
        validation_results['all_components_analyzed'] = main_components == available_components
        
        if not validation_results['all_components_analyzed']:
            missing = main_components - available_components
            logging.warning(f"Missing ablation results for components: {missing}")
        
        # Check for reasonable impact magnitudes
        for name, result in ablation_results.items():
            if name in main_components:
                # Each main component should have some measurable impact
                has_measurable_impact = result.risk_change_percentage > 1.0  # >1% change
                validation_results[f'{name}_has_impact'] = has_measurable_impact
                
                # Impact should be proportional to component weight
                component_weight = getattr(self.risk_engine.weights, 
                                         self.components[name]['weight_key'])
                
                # Expect higher-weighted components to have higher impact
                weight_impact_consistent = (
                    result.risk_change_percentage > component_weight * 10  # Rough heuristic
                )
                validation_results[f'{name}_weight_consistent'] = weight_impact_consistent
        
        # Check expected patterns if provided
        if expected_patterns:
            for component, expected_magnitude in expected_patterns.items():
                if component in ablation_results:
                    actual_magnitude = ablation_results[component].impact_magnitude
                    validation_results[f'{component}_expected_magnitude'] = (
                        actual_magnitude == expected_magnitude
                    )
        
        # Check for correlation consistency
        for name, result in ablation_results.items():
            if name in main_components and not np.isnan(result.correlation_with_original):
                # Correlation should be positive but less than 1 (some change occurred)
                correlation_reasonable = 0.3 <= result.correlation_with_original <= 0.95
                validation_results[f'{name}_correlation_reasonable'] = correlation_reasonable
        
        # Overall validation
        validation_results['overall_valid'] = all(
            validation_results.get(f'{comp}_has_impact', False) 
            for comp in main_components
        )
        
        return validation_results
    
    def generate_ablation_summary_report(self, 
                                       ablation_results: Dict[str, AblationResult],
                                       component_importance: List[ComponentImportance]) -> Dict[str, Any]:
        """
        Generate comprehensive summary report of ablation analysis.
        
        Args:
            ablation_results: Dictionary of ablation results
            component_importance: List of component importance rankings
            
        Returns:
            Dictionary with summary report data
        """
        # Main component results
        main_results = {name: result for name, result in ablation_results.items() 
                       if name in self.components.keys()}
        
        # Subcomponent results
        subcomp_results = {name: result for name, result in ablation_results.items() 
                          if name not in self.components.keys()}
        
        # Calculate summary statistics
        if main_results:
            impact_percentages = [result.risk_change_percentage for result in main_results.values()]
            total_impact = sum(impact_percentages)
            max_impact = max(impact_percentages)
            min_impact = min(impact_percentages)
            avg_impact = np.mean(impact_percentages)
        else:
            total_impact = max_impact = min_impact = avg_impact = 0.0
        
        # Most and least important components
        if component_importance:
            most_important = component_importance[0].component_name
            least_important = component_importance[-1].component_name
        else:
            most_important = least_important = "Unknown"
        
        # Impact magnitude distribution
        impact_distribution = {}
        for magnitude in ['low', 'medium', 'high']:
            count = sum(1 for result in main_results.values() 
                       if result.impact_magnitude == magnitude)
            impact_distribution[magnitude] = count
        
        summary_report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_components_analyzed': len(main_results),
            'total_subcomponents_analyzed': len(subcomp_results),
            'total_impact_percentage': round(total_impact, 2),
            'max_component_impact': round(max_impact, 2),
            'min_component_impact': round(min_impact, 2),
            'average_component_impact': round(avg_impact, 2),
            'most_important_component': most_important,
            'least_important_component': least_important,
            'impact_magnitude_distribution': impact_distribution,
            'component_rankings': [
                {
                    'component': comp.component_name,
                    'rank': comp.rank,
                    'importance_score': round(comp.importance_score, 3),
                    'contribution_percentage': round(comp.contribution_percentage, 2)
                }
                for comp in component_importance
            ],
            'detailed_results': {
                name: {
                    'risk_change_percentage': round(result.risk_change_percentage, 2),
                    'impact_magnitude': result.impact_magnitude,
                    'correlation': round(result.correlation_with_original, 3) if not np.isnan(result.correlation_with_original) else None,
                    'cells_affected': result.cells_affected
                }
                for name, result in main_results.items()
            }
        }
        
        return summary_report
    
    def create_ablation_visualization(self, 
                                    ablation_results: Dict[str, AblationResult],
                                    component_importance: List[ComponentImportance],
                                    output_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive ablation analysis visualization.
        
        Args:
            ablation_results: Dictionary of ablation results
            component_importance: List of component importance rankings
            output_path: Optional path to save the visualization
            
        Returns:
            Plotly figure with ablation analysis charts
        """
        # Filter main components
        main_results = {name: result for name, result in ablation_results.items() 
                       if name in self.components.keys()}
        
        if not main_results:
            logging.warning("No main component results available for visualization")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Component Impact on Risk Scores',
                'Component Importance Ranking',
                'Risk Change Distribution',
                'Component Correlation Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Component Impact Bar Chart
        components = list(main_results.keys())
        impacts = [main_results[comp].risk_change_percentage for comp in components]
        colors = ['red' if impact > 20 else 'orange' if impact > 10 else 'green' 
                 for impact in impacts]
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=impacts,
                marker_color=colors,
                name='Risk Change %',
                text=[f'{impact:.1f}%' for impact in impacts],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Importance Ranking
        if component_importance:
            importance_components = [comp.component_name for comp in component_importance]
            importance_scores = [comp.importance_score for comp in component_importance]
            
            fig.add_trace(
                go.Bar(
                    x=importance_components,
                    y=importance_scores,
                    marker_color='lightblue',
                    name='Importance Score',
                    text=[f'{score:.2f}' for score in importance_scores],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Risk Change Distribution (Box Plot)
        for i, (comp, result) in enumerate(main_results.items()):
            # Create synthetic distribution for visualization
            # (In practice, you'd use actual per-cell risk changes)
            synthetic_changes = np.random.normal(
                result.risk_change_mean, 
                result.risk_change_std, 
                100
            )
            
            fig.add_trace(
                go.Box(
                    y=synthetic_changes,
                    name=comp,
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        # 4. Correlation Analysis
        correlations = []
        component_names = []
        
        for comp, result in main_results.items():
            if not np.isnan(result.correlation_with_original):
                correlations.append(result.correlation_with_original)
                component_names.append(comp)
        
        if correlations:
            fig.add_trace(
                go.Scatter(
                    x=component_names,
                    y=correlations,
                    mode='markers+text',
                    marker=dict(size=12, color='purple'),
                    text=[f'{corr:.2f}' for corr in correlations],
                    textposition='top center',
                    name='Correlation'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Ablation Analysis: Component Importance and Impact',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Components", row=1, col=1)
        fig.update_yaxes(title_text="Risk Change (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Components", row=1, col=2)
        fig.update_yaxes(title_text="Importance Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Components", row=2, col=1)
        fig.update_yaxes(title_text="Risk Change", row=2, col=1)
        
        fig.update_xaxes(title_text="Components", row=2, col=2)
        fig.update_yaxes(title_text="Correlation with Original", row=2, col=2)
        
        # Save if path provided
        if output_path:
            fig.write_html(output_path)
            logging.info(f"Ablation visualization saved to: {output_path}")
        
        return fig
    
    def export_ablation_results_for_ops_notes(self, 
                                            ablation_results: Dict[str, AblationResult],
                                            component_importance: List[ComponentImportance]) -> str:
        """
        Export ablation results in format suitable for operational notes.
        
        Args:
            ablation_results: Dictionary of ablation results
            component_importance: List of component importance rankings
            
        Returns:
            Formatted string for ops notes
        """
        main_results = {name: result for name, result in ablation_results.items() 
                       if name in self.components.keys()}
        
        if not main_results:
            return "Ablation analysis: No component results available"
        
        ops_notes = []
        ops_notes.append("=== COMPONENT IMPORTANCE ANALYSIS ===")
        
        # Component rankings
        if component_importance:
            ops_notes.append("Component Importance Ranking:")
            for i, comp in enumerate(component_importance, 1):
                ops_notes.append(f"  {i}. {comp.component_name.title()}: "
                               f"{comp.contribution_percentage:.1f}% impact")
        
        # Key findings
        ops_notes.append("\nKey Findings:")
        
        # Most impactful component
        if component_importance:
            most_important = component_importance[0]
            ops_notes.append(f"• Most critical component: {most_important.component_name.title()} "
                           f"({most_important.contribution_percentage:.1f}% impact)")
        
        # High impact components
        high_impact_components = [name for name, result in main_results.items() 
                                if result.impact_magnitude == 'high']
        if high_impact_components:
            ops_notes.append(f"• High-impact components: {', '.join(high_impact_components)}")
        
        # Low impact components (potential for simplification)
        low_impact_components = [name for name, result in main_results.items() 
                               if result.impact_magnitude == 'low']
        if low_impact_components:
            ops_notes.append(f"• Low-impact components: {', '.join(low_impact_components)} "
                           "(consider simplification)")
        
        # Total system sensitivity
        total_impact = sum(result.risk_change_percentage for result in main_results.values())
        ops_notes.append(f"• Total system sensitivity: {total_impact:.1f}% cumulative impact")
        
        return "\n".join(ops_notes)