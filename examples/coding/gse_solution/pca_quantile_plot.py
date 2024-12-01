import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

def compute_dataset_statistics(datasets, after_normalization=False):
    """Compute and print mean and variance for each dataset."""
    for gse_id, data in datasets.items():
        mean_value = np.nanmean(data.values)
        variance_value = np.nanvar(data.values)
        print(f"{gse_id} - {'After Normalization - ' if after_normalization else ''}Mean: {mean_value:.2f}, Variance: {variance_value:.2f}")

def plot_data_distributions(datasets, after_normalization=False):
    """Plot data distributions for each dataset."""
    for gse_id, data in datasets.items():
        flattened_data = data.values.flatten()
        flattened_data = flattened_data[~np.isnan(flattened_data)]
        plt.figure(figsize=(10, 6))
        sns.histplot(flattened_data, bins=100, color='blue', alpha=0.7)
        plt.title(f"{'After Normalization - ' if after_normalization else ''}Data Distribution for {gse_id}")
        plt.xlabel('Expression Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'./output/{"after_" if after_normalization else ""}data_distribution_{gse_id}.png')
        plt.close()

def check_zero_variance_genes(data, gse_id, after_normalization=False):
    """Check for genes with zero variance across samples."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a Pandas DataFrame.")
    # Calculate variance across samples (axis=1)
    variances = data.var(axis=1)
    # Identify zero variance genes
    zero_variance_genes = variances == 0
    num_zero_variance = zero_variance_genes.sum()
    print(f"{gse_id} - {'After Normalization - ' if after_normalization else ''}Number of zero variance genes: {num_zero_variance}")

def quantile_normalize(df):
    """Perform Quantile Normalization on a DataFrame."""
    # Fill NaNs with the mean expression of each gene (row)
    df = df.apply(lambda x: x.fillna(x.mean()), axis=1)
    # Sort each column (sample)
    sorted_df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    # Compute the mean of each row (rank across samples)
    mean_ranks = sorted_df.mean(axis=1)
    # Get the ranks of the original data
    ranks = df.rank(method='min', axis=0)
    # Map the ranks to the mean ranks
    df_normalized = df.copy()
    for col in df.columns:
        rank_to_value = dict(zip(np.arange(1, len(mean_ranks)+1), mean_ranks))
        df_normalized[col] = ranks[col].map(rank_to_value)
    return df_normalized

def main():
    # List of GSE IDs to process
    gse_ids = ['GSE176043', 'GSE41781', 'GSE190986', 'GSE144600']

    datasets = {}
    output_dir = './output'
    if not os.path.exists(output_dir):
        print("Output directory does not exist. Please ensure that the processed data files are available.")
        return

    # Load each processed dataset
    for gse_id in gse_ids:
        processed_file = f'{output_dir}/{gse_id}_processed.csv'
        if os.path.exists(processed_file):
            print(f"Loading processed data for {gse_id} from {processed_file}")
            data = pd.read_csv(processed_file, index_col=0, low_memory=False)
            # Ensure all data is numeric except the index
            data = data.apply(pd.to_numeric, errors='coerce')
            datasets[gse_id] = data
        else:
            print(f"Processed data file for {gse_id} not found at {processed_file}.")
            print(f"Please ensure that the file exists or run the data processing script first.")
            return

    # Compute statistics before normalization
    print("\nStatistics before normalization:")
    compute_dataset_statistics(datasets)
    plot_data_distributions(datasets)

    # Check zero variance genes before normalization
    print("\nZero variance genes before normalization:")
    for gse_id, data in datasets.items():
        check_zero_variance_genes(data, gse_id)

    # Apply Quantile Normalization to each dataset
    datasets_normalized = {}
    for gse_id, data in datasets.items():
        print(f"\nApplying Quantile Normalization to {gse_id}")
        # Fill NaNs with the mean expression of each gene
        data = data.apply(lambda x: x.fillna(x.mean()), axis=1)
        data_normalized = quantile_normalize(data)
        datasets_normalized[gse_id] = data_normalized

    # Compute statistics after normalization
    print("\nStatistics after normalization:")
    compute_dataset_statistics(datasets_normalized, after_normalization=True)
    plot_data_distributions(datasets_normalized, after_normalization=True)

    # Check zero variance genes after normalization
    print("\nZero variance genes after normalization:")
    for gse_id, data in datasets_normalized.items():
        check_zero_variance_genes(data, gse_id, after_normalization=True)

    # Proceed with combining datasets
    gse_gene_sets = {}
    for gse_id, data in datasets_normalized.items():
        gene_set = set(data.index)
        gse_gene_sets[gse_id] = gene_set
        print(f"{gse_id} has {len(gene_set)} genes after normalization.")

    # Calculate pairwise intersections
    print("\nSanity Check: Pairwise Intersection Sizes of Gene Indices")
    for gse1, gse2 in combinations(gse_ids, 2):
        genes1 = gse_gene_sets[gse1]
        genes2 = gse_gene_sets[gse2]
        intersection_size = len(genes1 & genes2)
        union_size = len(genes1 | genes2)
        print(f"Intersection of {gse1} and {gse2}: {intersection_size} genes")
        print(f"Union of {gse1} and {gse2}: {union_size} genes")
        overlap_percentage = (intersection_size / union_size) * 100 if union_size > 0 else 0
        print(f"Overlap percentage between {gse1} and {gse2}: {overlap_percentage:.2f}%\n")

    # Combine the datasets on common genes (intersection)
    common_genes = set.intersection(*[set(data.index) for data in datasets_normalized.values()])
    if not common_genes:
        print("No common genes across all datasets after normalization.")
        return
    common_genes = sorted(common_genes)  # Convert set to sorted list
    print(f"Number of common genes across all datasets: {len(common_genes)}")

    all_data = []
    for gse_id in gse_ids:
        data = datasets_normalized.get(gse_id)
        if data is not None:
            # Subset to common genes
            data = data.loc[common_genes]
            # Transpose data so that samples are rows
            data_T = data.transpose()
            # Add a column for the GSE ID
            data_T['GSE_ID'] = gse_id
            # Add a column for the sample ID
            data_T['SampleID'] = data_T.index + f"_{gse_id}"
            all_data.append(data_T)
        else:
            print(f"No data available for {gse_id}")

    # Concatenate all data
    combined_data = pd.concat(all_data, axis=0)

    if combined_data.empty:
        print("Combined data is empty after concatenation.")
        return

    # Separate labels and sample IDs
    labels = combined_data['GSE_ID'].values
    sample_ids = combined_data['SampleID'].values
    combined_data = combined_data.drop(['GSE_ID', 'SampleID'], axis=1)

    # Ensure that all columns are numeric
    combined_data = combined_data.apply(pd.to_numeric, errors='coerce')

    # Remove columns (genes) with NaN values
    combined_data = combined_data.dropna(axis=1)

    # Remove rows (samples) with NaN values
    combined_data = combined_data.dropna(axis=0)

    if combined_data.empty:
        print("Combined data is empty after dropping NaN values.")
        return

    # Standardize the data
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
    print(f"\nExplained variance by PC1: {explained_variance[0]*100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1]*100:.2f}%")

if __name__ == '__main__':
    main()
