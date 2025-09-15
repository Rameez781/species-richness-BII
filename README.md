# Species Richness Prediction for Biodiversity Intactness Index Assessment Using Machine Learning

A machine learning analysis predicting global species richness patterns using environmental drivers from the PREDICTS database and WorldClim climate data. This research achieved exceptional predictive accuracy with R² = 0.994 using Gradient Boosting to model species diversity across 966,130 biodiversity records from 5,536 unique sites spanning 41 countries.

## Overview
This study implements multiple machine learning algorithms to predict the number of species found at different locations worldwide. The analysis combines biodiversity observations with climate variables to identify environmental factors that drive species diversity patterns. Distance from equator emerged as the dominant predictor followed by climate optimality indices and anthropogenic pressure measures.

## Data Sources
The research utilizes two primary datasets. The PREDICTS database provides global species abundance measurements from ecological studies conducted between 1993-2018, available at https://data.nhm.ac.uk/dataset/release-of-data-added-to-the-predicts-database-november-2022. Environmental predictors derive from WorldClim 2.1 bioclimatic variables at 10-arcminute spatial resolution covering temperature, precipitation, and seasonality patterns, accessible at https://www.worldclim.org/data/worldclim21.html.

## Methodology
The analysis pipeline combines PREDICTS biodiversity observations with WorldClim climate variables through coordinate-based spatial extraction. Feature engineering creates climate optimality indices, anthropogenic pressure measures, and biogeographic clustering variables. Model development compares traditional machine learning approaches including Random Forest and Gradient Boosting against deep learning architectures using rigorous cross-validation procedures. Stratified sampling ensures balanced representation across species richness gradients from 1 to 1,121 species per site.

## Key Results
Ten different algorithms were evaluated with Gradient Boosting achieving the highest accuracy followed by K-Nearest Neighbors and Random Forest. The study successfully identified critical thresholds where biodiversity decline accelerates under human pressure at values exceeding 0.6 on the anthropogenic pressure index. Climate optimality thresholds at 20°C temperature and 1500mm precipitation support maximum species richness across global ecosystems.

## Technical Implementation
The implementation uses Python 3 with scientific computing libraries including scikit-learn, PyTorch, pandas, and NumPy. Cross-validation employs 5-fold stratification to maintain consistent performance assessment while preventing overfitting through systematic hyperparameter optimization. The methodology provides quantitative tools for protected area optimization and restoration site selection.

## Repository Contents
The project contains the main analysis notebook with complete implementation, dissertation documentation, performance results and visualizations, and modular Python scripts for data processing and model training. Supporting files include requirements specifications, model evaluation metrics, and feature importance analyses demonstrating the relative influence of environmental variables on species diversity patterns.

## Usage
Run the main analysis notebook to reproduce all results and visualizations. The implementation requires pandas 1.3.0 minimum, numpy 1.21.0 minimum, scikit-learn 1.0.0 minimum, torch 1.9.0 minimum, matplotlib 3.4.0 minimum, and seaborn 0.11.0 minimum. Model training takes approximately 2-3 hours on standard computing hardware depending on the number of algorithms selected.

## Applications
Results support conservation planning applications through biodiversity hotspot identification and environmental impact assessment capabilities. The methodology can be extended to different geographic regions with appropriate calibration of climate optimality thresholds and anthropogenic pressure measures. Integration with remote sensing data streams could enable real-time biodiversity monitoring for conservation management.

## Citation
University of Reading MSc Data Science and Advanced Computing dissertation project, September 2025. PREDICTS database acknowledgment to Hudson et al. and WorldClim data attribution to Fick and Hijmans for enabling this research through open-access data sharing initiatives.
