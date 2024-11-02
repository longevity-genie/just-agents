import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

def normalize_dataset(data, gse_id):
    # Ensure that all column names (sample names) are strings
    data.columns = data.columns.astype(str)
    # Ensure that all index values (gene names) are strings
    data.index = data.index.astype(str)
    # Ensure that all data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Handle missing values
    # Drop columns (genes) that are all NaN
    data = data.dropna(axis=1, how='all')
    # Drop rows (samples) that are all NaN
    data = data.dropna(axis=0, how='all')

    # Set a threshold for maximum allowed NaNs
    # For example, drop columns (genes) with less than 50% valid data
    data = data.dropna(axis=1, thresh=int(data.shape[0] * 0.5))
    # Similarly, drop rows (samples) with less than 50% valid data
    data = data.dropna(axis=0, thresh=int(data.shape[1] * 0.5))

    # If data is empty after dropping, handle it
    if data.empty:
        print(f"{gse_id} data is empty after preprocessing. Skipping this dataset.")
        return None  # Return None to indicate skipping

    # Fill remaining NaNs with the mean of each gene
    data = data.fillna(data.mean())

    # Check if data contains negative values
    if (data.values < 0).any():
        print(f"{gse_id} contains negative values. Skipping log transformation.")
        # For datasets with negative values, skip log transformation
        pass
    else:
        # Add a small constant to avoid log(0)
        data = data + 1
        # Log2 transformation
        data = np.log2(data)

    # Z-score normalization
    data_transposed = data.transpose()
    # Ensure that columns (genes) are strings after transposition
    data_transposed.columns = data_transposed.columns.astype(str)
    # Ensure that index (samples) are strings after transposition
    data_transposed.index = data_transposed.index.astype(str)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_transposed)
    # Create DataFrame with the same index and columns
    data_scaled = pd.DataFrame(data_scaled, index=data_transposed.index, columns=data_transposed.columns)
    # Transpose back to original shape
    return data_scaled.transpose()


def plot_data_distributions(datasets):
    for gse_id, data in datasets.items():
        flattened_data = data.values.flatten()
        flattened_data = flattened_data[~np.isnan(flattened_data)]
        plt.figure(figsize=(10, 6))
        plt.hist(flattened_data, bins=100, color='blue', alpha=0.7)
        plt.title(f'Data Distribution for {gse_id}')
        plt.xlabel('Expression Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.show()

def compute_dataset_statistics(datasets):
    for gse_id, data in datasets.items():
        mean_value = np.nanmean(data.values)
        variance_value = np.nanvar(data.values)
        print(f"{gse_id} - Mean: {mean_value:.2f}, Variance: {variance_value:.2f}")

def check_zero_variance_genes(data, gse_id):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a Pandas DataFrame.")
    # Calculate variance across rows (genes)
    variances = data.var(axis=1)
    # Identify zero variance genes
    zero_variance_genes = variances == 0
    num_zero_variance = zero_variance_genes.sum()
    print(f"{gse_id} - Number of zero variance genes: {num_zero_variance}")

def main():
    # List of GSE IDs to process
    # gse_ids = ['GSE41781', 'GSE144600']  # Removed GSE148911
    gse_ids = ['GSE176043', 'GSE41781', 'GSE190986', 'GSE144600']
    datasets = {}
    output_dir = './output'

    for gse_id in gse_ids:
        processed_file = f'{output_dir}/{gse_id}_processed.csv'
        if os.path.exists(processed_file):
            print(f"Loading processed data for {gse_id} from {processed_file}")
            data = pd.read_csv(processed_file, index_col=0, low_memory=False)
            # Normalize the dataset
            data = normalize_dataset(data, gse_id)
            datasets[gse_id] = data
        else:
            print(f"Processed data file for {gse_id} not found at {processed_file}.")
            return

    # Combine the datasets
    all_data = []
    for gse_id in gse_ids:
        data = datasets.get(gse_id)
        if data is not None:
            # Transpose data so that samples are rows
            data_T = data.transpose()
            # Add a column for the GSE ID
            data_T['GSE_ID'] = gse_id
            # Add a column for the sample ID
            data_T['SampleID'] = data_T.index + f"_{gse_id}"
            all_data.append(data_T)
        else:
            print(f"No data available for {gse_id}")

    # Concatenate all data on common genes using an inner join
    combined_data = pd.concat(all_data, axis=0, join='inner')

    if combined_data.empty:
        print("Combined data is empty after intersection of genes.")
        return

    # Separate labels and sample IDs before converting data to numeric
    labels = combined_data['GSE_ID'].values
    sample_ids = combined_data['SampleID'].values
    combined_data = combined_data.drop(['GSE_ID', 'SampleID'], axis=1)

    # Ensure that all columns are numeric
    combined_data = combined_data.apply(pd.to_numeric, errors='coerce')

    # Remove genes (columns) and samples (rows) with NaN values
    combined_data = combined_data.dropna(axis=1, how='any')
    combined_data = combined_data.dropna(axis=0, how='any')

    if combined_data.empty:
        print("Combined data is empty after dropping NaN values.")
        return

    # Standardize the combined data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(combined_data)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['GSE_ID'] = labels
    principal_df['SampleID'] = sample_ids

    # Plot PCA with per-GSE mapping
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=principal_df,
        x='PC1', y='PC2',
        hue='GSE_ID',
        style='GSE_ID',
        s=100,
        palette='tab10'
    )
    plt.title('PCA of Datasets with Per-GSE Mapping')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    plt.legend(title='GSE ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./output/pca_plot_per_gse.png')
    plt.close()

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0]*100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1]*100:.2f}%")

if __name__ == '__main__':
    main()
