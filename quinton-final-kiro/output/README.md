# MISO Weather-Stress Heatmap Outputs

Generated on: 2025-10-10 10:53:27

## Directory Structure

### 📊 heatmaps/
Interactive weather-stress heatmaps for MISO territory:
- `miso_full_heatmap_12h.html` - 12-hour forecast heatmap
- `miso_full_heatmap_24h.html` - 24-hour forecast heatmap  
- `miso_full_heatmap_36h.html` - 36-hour forecast heatmap
- `miso_full_heatmap_48h.html` - 48-hour forecast heatmap

**Usage:** Open any HTML file in your web browser to view interactive maps.

### 📈 data/
Risk score data in CSV format:
- `risk_scores_12h.csv` - 12-hour risk data (all grid cells)
- `risk_scores_24h.csv` - 24-hour risk data (all grid cells)
- `risk_scores_36h.csv` - 36-hour risk data (all grid cells)  
- `risk_scores_48h.csv` - 48-hour risk data (all grid cells)

**Usage:** Import into analysis tools, Excel, or Python for further processing.

### 📋 reports/
Operational summaries and analysis reports:
- `operational_summary.txt` - Top risk areas and operational insights
- `ops_notes_*.txt` - Detailed operational notes by timestamp

**Usage:** Review for operational decision-making and risk assessment.

### 📝 logs/
System execution logs:
- `*.log` - Detailed system execution logs
- `pipeline.log` - Main pipeline execution log

**Usage:** Review for system performance and debugging.

### 📦 archive/
Archived files (old versions, test files):
- Previous heatmap versions
- Test outputs
- Development files

**Usage:** Reference only - current outputs are in other directories.

## Quick Start

1. **View Heatmaps:** Open `heatmaps/miso_full_heatmap_24h.html` in your browser
2. **Analyze Data:** Import `data/risk_scores_24h.csv` into your analysis tool
3. **Review Operations:** Read `reports/operational_summary.txt`

## System Information

- **Territory:** MISO Regional Transmission Organization
- **Grid Cells:** ~630 hexagonal cells with risk data
- **Risk Scale:** 0.0 (low) to 3.0+ (extreme)
- **Forecast Horizons:** 12, 24, 36, 48 hours
- **Data Sources:** Weather data, infrastructure capacity, population exposure
