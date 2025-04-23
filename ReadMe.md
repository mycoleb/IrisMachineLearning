<<<<<<< HEAD
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
=======
The project uses the famous Iris dataset to build and evaluate classification models. The dataset contains four features (sepal length, sepal width, petal length, and petal width) to classify three species of Iris flowers: setosa, versicolor, and virginica.


RandomForestClassifier's Role
The RandomForestClassifier plays a central role in this project as one of the main machine learning models used to classify iris flowers into different species. Here's how it's specifically used:

It's included in the ensemble of models being evaluated (alongside SVM and KNN)
It shows excellent performance and is selected as the "best model" to be saved for future use
It's used to analyze feature importance, helping identify which flower measurements are most useful for classification
The final prediction function relies on the saved RandomForest model to make species predictions on new flower measurements

Random Forest is particularly valuable in this project because:

It handles the multiclass nature of the problem well (classifying between three species)
It provides feature importance metrics, which adds interpretability
It's robust against overfitting with the small Iris dataset (150 samples)

Roles of Other Key Imports
Data Handling and Preprocessing

load_iris: Provides the classic Iris dataset with measured features and species labels
pandas: Creates a structured DataFrame for data manipulation and analysis
StandardScaler: Normalizes the feature values to have mean=0 and variance=1, improving model performance

Model Selection and Evaluation

train_test_split: Divides data into training and testing sets to properly evaluate model performance
cross_val_score: Performs k-fold cross-validation to get a robust estimate of model performance
Pipeline: Chains preprocessing (scaling) and model training into a single workflow
SVC (Support Vector Classifier): Provides an alternative classification approach that works well with clear decision boundaries
KNeighborsClassifier: Offers a simple, intuitive classification method based on proximity to known samples

Metrics and Visualization

classification_report, confusion_matrix, accuracy_score: Evaluate and quantify model performance
matplotlib and seaborn: Create visualizations of:

Feature distributions by species
Pairwise relationships between features
Feature correlations
Box plots showing data distributions
Confusion matrices illustrating classification results
Feature importance from the RandomForest model
Decision boundaries using PCA-reduced data



Dimensionality Reduction

PCA (Principal Component Analysis): Reduces the 4-dimensional feature space to 2 dimensions for visualization of decision boundaries

Model Persistence

joblib: Saves the trained model to disk, allowing it to be deployed in applications without retraining

Together, these components create a comprehensive workflow that demonstrates the entire machine learning pipeline from data exploration to model deployment for the iris flower classification task.
Data Analysis Insights
Looking at the visualizations:

Feature Distributions (Image 8): There's clear separation between species. Setosa has notably shorter petals than the other species, making it easily distinguishable.
Box Plots (Image 6):

Setosa has the shortest sepal and petal lengths
Virginica has the longest sepals and petals
Setosa has the widest sepals but narrowest petals


Pairplot (Image 9): Shows excellent separation of setosa from the other species across all features. Versicolor and virginica overlap somewhat but can still be distinguished.
Feature Correlation (Image 7):

Strong positive correlation (0.96) between petal length and petal width
Strong positive correlations between sepal length and both petal measurements
Slight negative correlation between sepal width and other measurements


Feature Importance (Image 5):

Petal width is the most important feature (about 0.45)
Petal length is the second most important (about 0.40)
Sepal measurements have much lower importance



Model Performance
Three classification models were evaluated:

Random Forest (Image 2):

Perfect classification of setosa
Misclassified 1 versicolor as virginica
Misclassified 4 virginica as versicolor


Support Vector Machine (Image 3):

Perfect classification of setosa
Misclassified 1 versicolor as virginica
Misclassified 2 virginica as versicolor
Overall best performance with 42/45 correct classifications


K-Nearest Neighbors (Image 1):

Perfect classification of setosa
Perfect classification of versicolor
Misclassified 4 virginica as versicolor
Tied with SVM for accuracy


Decision Boundaries (Image 4):

After PCA, shows clear separation of the three species
Blue region (setosa) is completely separate
Some boundary overlap between versicolor and virginica



Implementation Details
The Python implementation includes:

Data preparation:

Loading the Iris dataset
Feature standardization
Train-test splitting (70/30)


Model development:

Three different algorithms evaluated
Pipeline implementation with standardization
Cross-validation for robust evaluation


Visualization and analysis:

Distribution plots
Correlation analysis
Feature importance
Decision boundary visualization


Model deployment:

Saving the best model
Function for making new predictions



Key Observations

The Iris setosa is the most distinct species and can be perfectly classified by all models.
Versicolor and virginica have some overlap, particularly for virginica samples that get misclassified as versicolor.
Petal measurements are much more important than sepal measurements for classification.
The Support Vector Machine model appears to be the best performer among the three, though all models achieve >90% accuracy.

This project demonstrates good machine learning practices including data exploration, visualization, model evaluation, and deployment. The code includes well-documented steps and generates comprehensive visualizations to understand the dataset and model performance.
>>>>>>> 8486ed11ca89b98fcf8131093ab485528bdcf839
