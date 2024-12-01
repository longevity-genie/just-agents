import GEOparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

# Load the first dataset
file_path = '/input/GSE176043_family.soft.gz'
gse = GEOparse.get_GEO(filepath=file_path)

# Extract expression data
samples = gse.gsms
expression_data = {gsm: samples[gsm].table['VALUE'] for gsm in samples}
expression_df = pd.DataFrame(expression_data)

# Check for missing values and handle them
expression_df.dropna(inplace=True)

# Normalize the data
expression_df = (expression_df - expression_df.mean()) / expression_df.std()

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expression_df.T)

# Plot PCA
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA of GSE176043')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# Save the plot
output_dir = Path('/output')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'PCA_GSE176043.png')
plt.close()

print("PCA plot for GSE176043 saved.")