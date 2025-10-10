# MISO Weather-Stress Heatmap - Usage Guide

## Quick Start Commands

### 1. Generate Complete Analysis
```bash
# Run full pipeline (recommended)
python run_complete_pipeline.py
```
**Output**: Risk data, heatmaps, and reports for all forecast horizons

### 2. Create Interactive Heatmaps
```bash
# Generate heatmaps from existing data
python create_full_heatmap.py
```
**Output**: Interactive HTML heatmaps in `output/heatmaps/`

## Understanding the Outputs

### Interactive Heatmaps
- **Location**: `output/heatmaps/miso_full_heatmap_*.html`
- **Usage**: Open in web browser
- **Features**: Click cells for detailed risk information

### Risk Data
- **Location**: `output/data/risk_scores_*.csv`
- **Usage**: Import into Excel, Python, R for analysis
- **Columns**: cell_id, risk scores, weather data, confidence

### Operational Reports
- **Location**: `output/reports/operational_summary.txt`
- **Usage**: Review top risk areas and operational insights

## Risk Interpretation

### Risk Levels
- **0.0-0.6**: Low risk (green) - Normal operations
- **0.6-1.2**: Medium risk (yellow/orange) - Monitor conditions
- **1.2-2.4**: High risk (red) - Consider operational adjustments
- **2.4+**: Extreme risk (dark red) - Immediate attention required

### Risk Components
- **Hazard**: Weather-driven stress factors
- **Exposure**: Population and infrastructure at risk
- **Vulnerability**: System susceptibility to stress

## Troubleshooting

### Common Issues
1. **No heatmaps generated**: Check `output/logs/` for errors
2. **Missing data**: Verify internet connection for weather APIs
3. **Slow performance**: Reduce grid size or use demo mode

### Support
- Check logs in `output/logs/`
- Review system requirements in README.md
- Verify all dependencies are installed
