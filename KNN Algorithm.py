import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset from an Excel file
data_file = "us-violence-brief.xls"
data = pd.read_excel(data_file)

# Extract features (date and location) and target variable (fatalities)
features = data[['year', 'month', 'day', 'latitude', 'longitude']].values
labels = data['fatalities'].values

# Define the values of k (number of neighbors) to evaluate
k_values = [3, 5, 7, 9, 11]

# Initialize a list to store accuracy results for each k
accuracy_results = []

for k in k_values:
    # Separate training data (2020-2022)
    train_features = features[(features[:, 0] >= 2020) & (features[:, 0] <= 2022)]
    train_labels = labels[(features[:, 0] >= 2020) & (features[:, 0] <= 2022)]
    
    # Separate validation data (2023)
    validation_features = features[features[:, 0] == 2023]
    validation_labels = labels[features[:, 0] == 2023]
    
    # Create the k-nearest neighbors classifier
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier on the training data (2020-2022)
    knn_model.fit(train_features[:, 3:], train_labels)
    
    # Make predictions on the validation set (2023)
    predictions = knn_model.predict(validation_features[:, 3:])
    
    # Calculate the classification accuracy
    accuracy = accuracy_score(validation_labels, predictions)
    accuracy_results.append(accuracy)
    
    # Count the number of correctly and incorrectly classified data points
    correctly_classified = int(np.sum(predictions == validation_labels))
    misclassified = len(validation_labels) - correctly_classified
    
    # Compute the fraction of correctly classified data points
    fraction_correct = correctly_classified / len(validation_labels)
    
    # Print the results for the current value of k
    print(f"For k={k}, Accuracy: {accuracy}")
    print(f"Number of correctly classified 2023 demonstrations: {correctly_classified}")
    print(f"Number of misclassified 2023 demonstrations: {misclassified}")
    print(f"Fraction of correctly classified 2023 demonstrations: {fraction_correct:.2f}")
    print()

# Identify the value of k that gives the highest accuracy
optimal_k = k_values[np.argmax(accuracy_results)]
print(f"Best value of k: {optimal_k}")