# Species Richness Prediction Using Machine Learning

A comprehensive machine learning analysis to predict global species richness patterns using environmental drivers from the PREDICTS database and WorldClim climate data.

## Project Overview

This research implements multiple machine learning algorithms to predict species richness across global ecosystems. The study analyzes 966,130 biodiversity records from 5,536 unique sites spanning 41 countries to identify environmental factors that drive species diversity patterns. The main objective is to determine how climate variables, anthropogenic pressure, and spatial factors influence the number of species found at different locations worldwide.

## Key Findings

The Gradient Boosting algorithm achieved exceptional predictive accuracy with R² = 0.994 and RMSE = 13.68 species, substantially outperforming traditional statistical approaches. Distance from equator emerged as the dominant predictor followed by climate optimality indices and anthropogenic pressure measures. The study successfully identified critical thresholds where biodiversity decline accelerates under human pressure.

## Implementation Details

### Methodology

The analysis pipeline consists of several integrated components. Data preprocessing combines PREDICTS biodiversity observations with WorldClim 2.1 bioclimatic variables through coordinate-based spatial extraction. Feature engineering creates climate optimality indices, anthropogenic pressure measures, and biogeographic clustering variables. Model development compares traditional machine learning approaches against deep learning architectures using rigorous cross-validation procedures.

### Technical Specifications

The implementation uses Python 3 with scientific computing libraries including scikit-learn, PyTorch, pandas, and NumPy. Stratified sampling ensures balanced representation across species richness gradients from 1 to 1,121 species per site. Cross-validation employs 5-fold stratification to maintain consistent performance assessment while preventing overfitting through systematic hyperparameter optimization.

## Repository Structure

The project contains the main analysis notebook with complete implementation, dissertation documentation, performance results and visualizations, and modular Python scripts for data processing and model training. Supporting files include requirements specifications, model evaluation metrics, and feature importance analyses.

## Data Sources

Biodiversity data comes from the PREDICTS database containing global species abundance measurements from ecological studies conducted between 1993-2018. Environmental predictors derive from WorldClim 2.1 bioclimatic variables at 10-arcminute spatial resolution covering temperature, precipitation, and seasonality patterns. The integrated dataset represents comprehensive geographic coverage across major biomes and land use categories.

## Model Performance

Ten different algorithms were evaluated including ensemble methods, support vector machines, neural networks, and transformer architectures. Gradient Boosting achieved the highest accuracy followed by K-Nearest Neighbors and Random Forest. Deep learning models showed competitive performance but did not exceed traditional ensemble methods for this ecological prediction task.

## Results and Applications

The analysis successfully identified key environmental drivers of species richness including latitudinal gradients, climate optimality thresholds, and anthropogenic impact boundaries. Results support conservation planning applications through biodiversity hotspot identification and environmental impact assessment capabilities. The methodology provides quantitative tools for protected area optimization and restoration site selection.

## Usage Instructions

Run the main analysis notebook to reproduce all results and visualizations. The implementation requires no external data downloads as feature engineering recreates all necessary variables from base environmental datasets. Model training takes approximately 2-3 hours on standard computing hardware depending on the number of algorithms selected.

## Validation and Robustness

Comprehensive validation procedures include cross-validation consistency assessment, overfitting analysis, and performance stability across different species richness ranges. Statistical significance testing confirms meaningful differences between algorithm performances while robustness scores integrate multiple evaluation criteria including generalization capability and prediction accuracy.

## Scientific Contributions

This research advances ecological modeling through rigorous data leakage prevention, systematic algorithm comparison, and integration of climate theory with computational methods. The study provides the first comprehensive evaluation of deep learning approaches for global species richness prediction while establishing new benchmarks for biodiversity assessment accuracy.

## Future Applications

The methodology can be extended to different geographic regions with appropriate calibration of climate optimality thresholds and anthropogenic pressure measures. Integration with remote sensing data streams could enable real-time biodiversity monitoring while incorporation of functional trait databases would expand beyond taxonomic diversity measures.

## Technical Requirements

Python 3.7 or higher with pandas 1.3.0 minimum, numpy 1.21.0 minimum, scikit-learn 1.0.0 minimum, torch 1.9.0 minimum, matplotlib 3.4.0 minimum, and seaborn 0.11.0 minimum. Additional dependencies include scipy for statistical analysis and plotly for interactive visualizations.

## Citation and Attribution

University of Reading MSc Data Science and Advanced Computing dissertation project, September 2025. PREDICTS database acknowledgment to Hudson et al. and WorldClim data attribution to Fick and Hijmans for enabling this research through open-access data sharing initiatives.
