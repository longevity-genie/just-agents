import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load the GSE41781 dataset
file_path = Path('./input/GSE41781_family.soft.gz')
gse41781 = GEOparse.get_GEO(filepath=str(file_path), silent=True)

# Extract the expression data
expression_data = gse41781.pivot_samples('VALUE')

# Fill missing values with the mean of each column
expression_data_filled = expression_data.fillna(expression_data.mean())

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(expression_data_filled)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA of GSE41781')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Save the plot
output_path = Path('./output/GSE41781_PCA_plot.png')
plt.savefig(output_path)
plt.close()

# Confirm the plot was saved
output_path.exists()