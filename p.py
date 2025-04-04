# Iris Flower Species Classifier
# A complete ML workflow including data visualization and model development

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier data manipulation
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Print basic information about the dataset
print("Dataset Information:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(feature_names)}")
print(f"Number of classes: {len(target_names)}")
print(f"Classes: {target_names}")
print("\nFeature Names:")
for name in feature_names:
    print(f"  - {name}")

# 2. Data Exploration and Visualization

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2.1 Distribution of features by species
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    for species in target_names:
        plt.hist(df[df['species'] == species][feature], alpha=0.7, bins=15, label=species)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'Distribution of {feature} by Species')
plt.tight_layout()
plt.savefig('iris_feature_distributions.png')

# 2.2 Pairplot to visualize relationships between features
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', markers=['o', 's', 'D'], palette='viridis')
plt.suptitle('Pairwise Relationships between Features', y=1.02)
plt.savefig('iris_pairplot.png')

# 2.3 Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.drop('species', axis=1).corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, mask=mask, linewidths=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png')

# 2.4 Box plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df, palette='viridis')
    plt.title(f'Box Plot of {feature} by Species')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('iris_boxplots.png')

# 3. Data Preprocessing and Model Training

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 3.1 Define models to evaluate
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# 3.2 Evaluate models using cross-validation
print("\nCross-Validation Results:")
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
        ('classifier', model)
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"{name}:")
    print(f"  - Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"  - Standard Deviation: {cv_scores.std():.4f}")
    
    # Train the model on the full training set
    pipeline.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  - Test Accuracy: {test_accuracy:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')

# 4. Feature Importance (for Random Forest)
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf = rf_pipeline.named_steps['classifier']

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nFeature Importances (Random Forest):")
for i in range(X.shape[1]):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 5. Visualizing Decision Boundaries (for best model)
# We'll use SVM for this visualization
from sklearn.decomposition import PCA

# Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train SVM on PCA-transformed data
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_pca, y)

# Create mesh grid for decision boundary visualization
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Plot the decision boundary
X0, X1 = X_pca[:, 0], X_pca[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure(figsize=(10, 8))
ax = plt.subplot(1, 1, 1)
plot_contours(ax, svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training points
for i, species in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                label=species, edgecolors='k')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Decision Boundaries (SVM) with PCA')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('decision_boundaries.png')

# 6. Save the best model
import joblib

best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
best_model.fit(X, y)  # Fit on the entire dataset

# Save the model
joblib.dump(best_model, 'iris_classifier_model.pkl')

# 7. Function to make predictions on new data
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict the iris species based on the four measurements.
    
    Parameters:
    - sepal_length: Sepal length in cm
    - sepal_width: Sepal width in cm
    - petal_length: Petal length in cm
    - petal_width: Petal width in cm
    
    Returns:
    - Predicted species name and probability
    """
    # Load the model
    model = joblib.load('iris_classifier_model.pkl')
    
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    species_id = model.predict(input_data)[0]
    species_name = target_names[species_id]
    
    # Get probability
    probabilities = model.predict_proba(input_data)[0]
    probability = probabilities[species_id]
    
    return {
        'species': species_name,
        'probability': probability,
        'all_probabilities': {
            target_names[i]: prob for i, prob in enumerate(probabilities)
        }
    }

# Test the prediction function with a sample
sample_prediction = predict_species(5.1, 3.5, 1.4, 0.2)  # Should be setosa
print("\nSample Prediction:")
print(f"Species: {sample_prediction['species']}")
print(f"Probability: {sample_prediction['probability']:.4f}")
print("All Probabilities:")
for species, prob in sample_prediction['all_probabilities'].items():
    print(f"  - {species}: {prob:.4f}")

print("\nIris Flower Species Classifier is ready for use!")