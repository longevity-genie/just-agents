import pandas as pd
import os
import GEOparse
import gzip
import urllib.request

def download_platform_annotation(platform_id):
    """Download the platform annotation file for a given platform ID."""
    platform_file = f'./input/{platform_id}.annot.gz'
    if not os.path.exists(platform_file):
        print(f"Downloading platform annotation file for {platform_id}")
        platform_url = f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{platform_id[:-3]}nnn/{platform_id}/annot/{platform_id}.annot.gz'
        try:
            urllib.request.urlretrieve(platform_url, platform_file)
        except Exception as e:
            print(f"Failed to download platform annotation for {platform_id}: {e}")
            return None
    else:
        print(f"Platform annotation file for {platform_id} already exists.")
    return platform_file

def parse_platform_annotation(platform_file):
    """Parse the platform annotation file and return a DataFrame."""
    try:
        with gzip.open(platform_file, 'rt', errors='ignore') as f:
            # Read lines until we reach the table header
            for line in f:
                if line.startswith('#'):
                    continue
                elif line.startswith('ID\t'):
                    header = line.strip().split('\t')
                    break
            else:
                raise ValueError("Platform annotation file does not contain header line starting with 'ID\t'")
            # Read the rest of the file into a DataFrame
            platform_table = pd.read_csv(f, sep='\t', names=header, comment='#', low_memory=False)
        return platform_table
    except Exception as e:
        print(f"Failed to parse platform annotation file: {e}")
        return None

def map_ids_to_symbols(gene_ids, platform_table, id_column='ID', symbol_column='Gene symbol'):
    """Map gene IDs to gene symbols using the platform annotation table."""
    # Ensure IDs are strings and strip whitespaces
    gene_ids = [str(gid).strip() for gid in gene_ids]
    platform_table[id_column] = platform_table[id_column].astype(str).str.strip()
    platform_table[symbol_column] = platform_table[symbol_column].astype(str).str.strip()

    # Create mapping dictionary, excluding entries with empty gene symbols
    id_to_symbol = {
        row[id_column]: row[symbol_column]
        for _, row in platform_table.iterrows()
        if row[symbol_column] and row[symbol_column] != 'NA'
    }

    # Map IDs to symbols
    mapping = {gene_id: id_to_symbol.get(gene_id) for gene_id in gene_ids}
    return mapping

def handle_duplicate_indices(df):
    """Handle duplicate gene symbols by taking the mean."""
    if df.index.duplicated().any():
        print("Duplicate gene symbols found. Aggregating duplicates by taking the mean.")
        df = df.groupby(level=0).mean()
    return df

def process_GSE(gse_id):
    """Process a GSE dataset and map gene identifiers to gene symbols."""
    print(f"\nProcessing {gse_id}")
    gse = GEOparse.get_GEO(geo=gse_id, destdir='./input', silent=True)
    all_expression_data = []  # Collect mapped data only
    mapped_genes = set()
    mapped_count = 0

    # Process data from GSM tables
    print(f"Processing data from GSM tables for {gse_id}")
    platform_ids = list(gse.gpls.keys())
    for platform_id in platform_ids:
        print(f"Platform ID: {platform_id}")
        platform_file = download_platform_annotation(platform_id)
        if platform_file is None:
            continue
        platform_table = parse_platform_annotation(platform_file)
        if platform_table is None:
            continue

        # Use a fixed symbol column
        symbol_column = 'Gene symbol'

        if symbol_column not in platform_table.columns:
            print(f"Symbol column '{symbol_column}' not found in platform annotation for {platform_id}.")
            continue

        # Extract expression data for this platform
        platform_data_frames = []
        for gsm_name, gsm in gse.gsms.items():
            if gsm.metadata['platform_id'][0] == platform_id:
                df = gsm.table[['ID_REF', 'VALUE']].set_index('ID_REF')
                df.columns = [gsm_name]
                platform_data_frames.append(df)
        if not platform_data_frames:
            print(f"No GSM data found for platform {platform_id} in {gse_id}.")
            continue
        platform_expression_data = pd.concat(platform_data_frames, axis=1)
        platform_expression_data.index = platform_expression_data.index.astype(str).str.strip()
        platform_table['ID'] = platform_table['ID'].astype(str).str.strip()
        platform_table[symbol_column] = platform_table[symbol_column].astype(str).str.strip()

        # Map IDs to gene symbols
        gene_ids = platform_expression_data.index.tolist()
        mapping = map_ids_to_symbols(gene_ids, platform_table, id_column='ID', symbol_column=symbol_column)

        # Apply mapping
        platform_expression_data.rename(index=mapping, inplace=True)

        # Replace NaN indices with original gene IDs
        platform_expression_data.index = pd.Index([
            original_id if pd.isnull(mapped_symbol) else mapped_symbol
            for original_id, mapped_symbol in zip(gene_ids, platform_expression_data.index)
        ]).astype(str)

        platform_expression_data.index.name = 'GeneSymbol_or_ID'

        # Handle duplicates
        platform_expression_data = handle_duplicate_indices(platform_expression_data)

        # Update counts
        mapped_genes.update(platform_expression_data.index)
        mapped_count += len(platform_expression_data.index)

        # Append the mapped data to the overall data list
        all_expression_data.append(platform_expression_data)

    if not all_expression_data:
        print(f"No data could be processed for {gse_id}.")
        return None, 0, 0

    # Combine data frames from all platforms
    combined_data = pd.concat(all_expression_data, axis=1)
    print(f"Number of genes in {gse_id} after mapping: {len(mapped_genes)}")
    print(f"First few gene symbols in {gse_id}: {list(mapped_genes)[:25]}")
    return combined_data, mapped_count, 0  # unmapped_count is 0 as we do not track unmapped genes here

def main():
    # Create input directory if it doesn't exist
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # List of GSE IDs to process
    gse_ids = ['GSE176043', 'GSE41781']

    # Process each GSE
    for gse_id in gse_ids:
        data, mapped_count, unmapped_count = process_GSE(gse_id)
        if data is not None:
            print(f"Processed {gse_id}: Mapped genes: {mapped_count}, Unmapped genes: {unmapped_count}")
            # Ensure the index is of type string
            data.index = data.index.astype(str)
            # Remove any duplicate columns that might have been added accidentally
            data = data.loc[:, ~data.columns.duplicated()]
            # Optionally, save the processed data to a file
            output_file = f'./output/{gse_id}_processed.csv'
            if not os.path.exists('./output'):
                os.makedirs('./output')
            # Save the DataFrame, ensuring the index is saved correctly
            data.to_csv(output_file, index=True)
            print(f"Processed data for {gse_id} saved to {output_file}")
        else:
            print(f"Failed to process {gse_id}")

if __name__ == '__main__':
    main()
