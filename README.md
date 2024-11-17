# KNN Violent Demonstrations Fatality Prediction

## Project Overview
This project uses **k-nearest neighbors (KNN)** classification to predict fatalities in violent demonstrations in the U.S. in 2023 based on historical data from 2020-2022. The dataset includes records of violent demonstrations with attributes such as the **date** (year, month, day) and **location** (latitude, longitude). The goal is to classify whether each event in 2023 resulted in fatalities and evaluate the model’s accuracy.

### Objective
The project focuses on:
- **KNN Classification**: To predict whether a demonstration had fatalities.
- **Cross-Validation**: Using K-fold cross-validation to evaluate the classifier.
- **Geospatial and Temporal Analysis**: Visualizing fatalities geographically and by month.
- **ROC Curve**: Plotting the Receiver Operating Characteristic (ROC) curve for model evaluation.

## Dataset
The dataset used for this analysis consists of violent demonstrations recorded by the **Armed Conflict Location & Event Data Project (ACLED)** from January 2020 to January 2024. Each demonstration is labeled with a binary target variable (`fatalities`) indicating whether fatalities occurred (`1` for fatalities, `0` for no fatalities).

## Key Features
- **KNN Algorithm**: For classification of fatalities.
- **Cross-Validation**: Using K-fold validation for model performance.
- **Geospatial Mapping**: Visualizing the decision boundaries on the U.S. map.
- **Temporal Analysis**: Analyzing fatalities by month.
- **ROC Curve**: Evaluating model performance.

## Installation
To run this project, you need Python and the following libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `mpl_toolkits.basemap`
- `geopy`

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib basemap geopy
