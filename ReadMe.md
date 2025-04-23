# Iris Flower Species Classifier

This repository contains a complete Machine Learning workflow aimed at classifying species of the Iris flower based on several measurements. The project uses Python with libraries like Scikit-learn, Pandas, NumPy, and Matplotlib/Seaborn for data visualization.

## Project Overview

The Iris Flower Species Classifier is built to predict the species of an Iris flower using its sepal and petal dimensions:
- Sepal length
- Sepal width
- Petal length
- Petal width

### Features

- Data loading and preprocessing
- Exploratory data analysis including histograms, pair plots, and box plots
- Building and evaluating models using Random Forest, SVM, and K-Nearest Neighbors
- Feature importance analysis
- Decision boundaries visualization with PCA-reduced data
- Model persistence for future predictions

### Visualizations Included

- **Feature Distributions by Species:** Histograms showing the distribution of each feature among the Iris species.
- **Pairplot:** Visualizing pairwise relationships between features to understand the interactions between different attributes.
- **Correlation Heatmap:** Analyzing feature correlations.
- **Box Plots:** Visualizing statistical summaries of various attributes.
- **Decision Boundaries:** Displaying how the SVM model classifies different species along the principal components.

### Model Performance

The models' performances are evaluated through cross-validation accuracy scores, confusion matrices, and classification reports, ensuring robust performance before deployment.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

### Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/mycoleb/IrisMachineLearning.git
cd IrisMachineLearning
