
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load iris dataset

iris = load_iris()


X = iris.data

y = iris.target

# Explore the dataset

print("Feature names:", iris.feature_names)

print("Target names:", iris.target_names)

print("First 5 samples:\n", X[:5])
 


# Scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset - Colored by Species")
plt.show()

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)
#Evaluate the Model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred, target_names=iris.target_names))


#########PARTB###########

#Applying K-means clustering
kmeans= KMeans(n_cluster=3)
kmeans.fit(X)
clusters= kmeans.predict(X)

# Compare with true labels(Visualize clustering results)
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel(clusters.feature_name[0])
plt.ylabel(clusters.feature_name[1])
plt.title("K-means Clustering Results")
plt.show()
