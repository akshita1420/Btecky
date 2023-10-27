# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Basic statistics of the dataset
print(iris.describe())

# Pairplot for visualization
sns.pairplot(iris, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.subplot(1, 2, 2)
sns.boxplot(x='species', y='sepal_width', data=iris)
plt.show()

# Correlation heatmap
correlation_matrix = iris.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()
