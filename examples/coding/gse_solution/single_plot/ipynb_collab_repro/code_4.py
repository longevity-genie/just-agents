import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load the normalized data for GSE176043
file_path = Path('./input/GSE176043_normalized.csv')
data = pd.read_csv(file_path)

# Extract the data for PCA, excluding the 'ID_REF' column
expression_data = data.drop(columns=['ID_REF']).values

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(expression_data)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA of GSE176043')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Save the plot
output_path = Path('./output/GSE176043_PCA_plot.png')
plt.savefig(output_path)
plt.close()

# Confirm the plot was saved
output_path.exists()