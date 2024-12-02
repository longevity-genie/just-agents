import GEOparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load GSE176043 again
print('Loading GSE176043 for PCA analysis...')
gse = GEOparse.get_GEO('GSE176043', destdir='/input')

# Extract expression data
if gse.gsms:
    gsm_data = gse.pivot_samples('VALUE')
    print(f'Loaded GSE176043 with shape: {gsm_data.shape}')
else:
    print('No GSM data found for GSE176043.')

# Normalize the data
print('Normalizing the data...')
scaler = StandardScaler()
normalized_data = scaler.fit_transform(gsm_data.T)  # Transpose to have samples as rows

# Perform PCA
print('Performing PCA...')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Plotting PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA of GSE176043')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

# Save the plot
output_path = '/output/pca_gse176043.png'
plt.savefig(output_path)
plt.close()
print(f'PCA plot saved to {output_path}')