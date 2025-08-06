# Exercise 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

print("All packages imported successfully!")

# Exercise 2 Part A: Supervised Learning - Logistic Regression
iris = load_iris()
X = iris.data
y = iris.target

# Show sample data
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("\nSample Data:")
print(df.head())

# Split data for Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred_log = log_model.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names))

# Visualize some features
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset - True Labels")
plt.show()

# Exercise 2 Part B: Unsupervised Learning - KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Visualize clustering results
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("KMeans Clustering Results")
plt.show()

# Exercise 3: Machine Learning Pipeline using KNN
# Re-split for KNN (different test size for variation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred_knn = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"\nKNN Model Accuracy: {accuracy:.2f}")
