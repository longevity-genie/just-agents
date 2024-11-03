import pandas as pd
import os
import GEOparse
import mygene
import gzip
import urllib.request
import scanpy as sc
from pybiomart import Server

# Initialize MyGeneInfo
mg = mygene.MyGeneInfo()
def get_mouse_to_human_gene_mapping():
    """Retrieve mapping from mouse Ensembl gene IDs to human gene symbols."""
    from pybiomart import Server
    print("Retrieving mouse to human gene mapping...")
    server = Server(host='http://www.ensembl.org')
    mart = server['ENSEMBL_MART_ENSEMBL']
    mouse_dataset = mart['mmusculus_gene_ensembl']

    # Attributes to retrieve
    attributes = [
        'ensembl_gene_id',
        'external_gene_name',
        'hsapiens_homolog_ensembl_gene',
        'hsapiens_homolog_associated_gene_name',
    ]

    # Query the dataset
    results = mouse_dataset.query(attributes=attributes)

    # Drop rows without human homologs
    results_non_nan = results.dropna(subset=['Human gene stable ID', 'Human gene name'], how='all')

    # Remove duplicates
    results_non_nan = results_non_nan.drop_duplicates(subset=['Gene stable ID'])

    # Create mapping dictionaries
    ensembl_to_human_symbol = dict(zip(results_non_nan['Gene stable ID'], results_non_nan['Human gene name']))
    mouse_symbol_to_human_symbol = dict(zip(results_non_nan['Gene name'], results_non_nan['Human gene name']))

    return ensembl_to_human_symbol, mouse_symbol_to_human_symbol

ensembl_to_human_symbol, mouse_symbol_to_human_symbol = get_mouse_to_human_gene_mapping()

def get_species(gse):
    """Determine the species of the GSE dataset."""
    # Try to get organism from GSE metadata
    organism = gse.metadata.get('organism', [''])[0].lower()
    if 'homo sapiens' in organism:
        return 'human'
    elif 'mus musculus' in organism:
        return 'mouse'
    else:
        # If not found in GSE metadata, check GSM samples
        for gsm in gse.gsms.values():
            organism = gsm.metadata.get('organism_ch1', [''])[0].lower()
            if 'homo sapiens' in organism:
                return 'human'
            elif 'mus musculus' in organism:
                return 'mouse'
        # Default to human if species cannot be determined
        print("Warning: Species could not be determined. Defaulting to human.")
        return 'human'


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

def download_file(url, dest_path):
    """Download a file from a URL to a specified destination path."""
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File {dest_path} already exists.")


def process_GSE144600():
    """Process GSE144600 and handle H5 files."""
    print("\nProcessing GSE144600")
    gse_id = 'GSE144600'
    supp_dir = f'./input/{gse_id}_supp'
    if not os.path.exists(supp_dir):
        os.makedirs(supp_dir)

    # Determine the species (should be 'mouse')
    # Since we don't have GSE metadata here, we'll set species directly
    species = 'mouse'
    is_mouse = species == 'mouse'
    print(f"Species for {gse_id}: {species}")

    # Retrieve mouse-to-human mapping
    if is_mouse:
        get_mouse_to_human_gene_mapping()

    # URLs for the supplementary files
    urls = {
        'minusDox': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144600/suppl/GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5',
        'plusDox': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144600/suppl/GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5'
    }

    # Download the H5 files
    h5_files = {}
    for key, url in urls.items():
        dest_path = os.path.join(supp_dir, os.path.basename(url))
        download_file(url, dest_path)
        h5_files[key] = dest_path

    data_frames = []
    mapped_genes = set()
    unmapped_count = 0
    mapped_count = 0

    for condition, h5_file in h5_files.items():
        print(f"Reading data from {h5_file}")
        try:
            # Read the H5 file using Scanpy
            adata = sc.read_10x_h5(h5_file)
            # Make variable names unique
            adata.var_names_make_unique()
            # Convert to DataFrame with cells as rows and genes as columns
            expression_data = pd.DataFrame(
                data=adata.X.toarray(),
                index=[f"{condition}_{cell}" for cell in adata.obs_names],
                columns=adata.var_names
            )
            # Transpose to have genes as index
            expression_data = expression_data.transpose()
            expression_data.index = expression_data.index.astype(str).str.strip()

            # Map mouse genes to human gene symbols
            gene_ids = expression_data.index.tolist()
            mapped_indices = [
                ensembl_to_human_symbol.get(gid, mouse_symbol_to_human_symbol.get(gid, None))
                for gid in gene_ids
            ]
            expression_data.index = mapped_indices

            # Count unmapped genes
            unmapped_genes_condition = expression_data.index.isnull().sum()
            print(f"Number of unmapped genes in {gse_id} - condition {condition}: {unmapped_genes_condition}")

            # Drop unmapped genes
            expression_data = expression_data[~expression_data.index.isnull()]

            # Handle duplicates
            expression_data = handle_duplicate_indices(expression_data)
            data_frames.append(expression_data)

            # Update counts
            mapped_genes.update(expression_data.index)
            mapped_count += len(expression_data.index)
            unmapped_count += unmapped_genes_condition
        except Exception as e:
            print(f"Failed to read {h5_file}: {e}")
            continue

    if not data_frames:
        print(f"No data could be extracted from H5 files for {gse_id}.")
        return None, 0, 0

    # Combine data from both conditions
    combined_data = pd.concat(data_frames, axis=1)
    combined_data.index.name = 'GeneSymbol_or_ID'

    print(f"Number of genes in {gse_id} after mapping: {len(mapped_genes)}")
    print(f"First few gene symbols in {gse_id}: {list(mapped_genes)[:25]}")

    return combined_data, mapped_count, unmapped_count


def process_GSE(gse_id):
    """Process a GSE dataset and map gene identifiers to gene symbols."""
    if gse_id == 'GSE144600':
        return process_GSE144600()
    print(f"\nProcessing {gse_id}")
    gse = GEOparse.get_GEO(geo=gse_id, destdir='./input', silent=True)
    all_expression_data = []  # Collect mapped data only
    mapped_genes = set()
    unmapped_count = 0
    mapped_count = 0

    # Determine the species
    species = get_species(gse)
    is_mouse = species == 'mouse'
    print(f"Species for {gse_id}: {species}")

    # Retrieve mouse-to-human mapping if needed
    if is_mouse:
        get_mouse_to_human_gene_mapping()

    # Check if GSM tables are empty and handle supplementary files
    first_gsm = next(iter(gse.gsms.values()))
    if first_gsm.table.empty:
        print(f"GSM tables for {gse_id} are empty. Attempting to process supplementary files.")
        supp_dir = f'./input/{gse.get_accession()}_supp'
        if not os.path.exists(supp_dir):
            os.makedirs(supp_dir)
        print(f"Downloading supplementary files for {gse_id} into {supp_dir}")
        try:
            gse.download_supplementary_files(supp_dir)
        except Exception as e:
            print(f"Failed to download supplementary files for {gse_id}: {e}")
            # Check for manually downloaded files
            if not os.listdir(supp_dir):
                print(f"No supplementary files found for {gse_id} in {supp_dir}.")
                manual_supp_dir = f'./input/{gse_id}_manual_supp'
                if os.path.exists(manual_supp_dir):
                    print(f"Using manually downloaded supplementary files from {manual_supp_dir}")
                    supp_dir = manual_supp_dir
                else:
                    print(f"No manual supplementary files found for {gse_id}. Skipping.")
                    return None, 0, 0

        # Process supplementary files
        data_frames = []  # Collect data from supplementary files
        for root, dirs, files in os.walk(supp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    if file.endswith('.gz'):
                        with gzip.open(file_path, 'rt', errors='ignore') as f:
                            df = pd.read_csv(f, sep='\t', index_col=0)
                    else:
                        df = pd.read_csv(file_path, sep='\t', index_col=0)
                    df.index = df.index.astype(str).str.strip()
                    df = handle_duplicate_indices(df)
                    data_frames.append(df)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    continue

        if not data_frames:
            print(f"No data could be extracted from supplementary files for {gse_id}.")
            return None, 0, 0

        # Combine data from supplementary files
        expression_data = pd.concat(data_frames, axis=1)


        # Map gene identifiers to gene symbols
        print(f"Mapping gene identifiers for {gse_id}")
        gene_ids = expression_data.index.tolist()
        # Remove version numbers if present
        gene_ids = [gid.split('.')[0] for gid in gene_ids]

        if is_mouse:
            # Map mouse Ensembl IDs or symbols to human gene symbols
            mapped_indices = [
                ensembl_to_human_symbol.get(gid, mouse_symbol_to_human_symbol.get(gid, None))
                for gid in gene_ids
            ]
        else:
            # For human datasets, use MyGeneInfo or keep the gene IDs as is
            id_type = 'ensembl.gene' if gene_ids[0].startswith('ENSG') else 'symbol'
            batch_size = 1000
            mapping = {}
            for i in range(0, len(gene_ids), batch_size):
                batch_ids = gene_ids[i:i+batch_size]
                try:
                    query = mg.querymany(batch_ids, scopes=id_type, fields='symbol', species=species, returnall=False, as_dataframe=True)
                    # Handle duplicate hits by selecting the first symbol
                    batch_mapping = {}
                    for gid in batch_ids:
                        try:
                            hits = query.loc[gid]
                            if isinstance(hits, pd.DataFrame):
                                # Multiple hits, select the first
                                symbol = hits.iloc[0]['symbol']
                            else:
                                symbol = hits['symbol']
                            batch_mapping[gid] = symbol
                        except KeyError:
                            # No hit found
                            batch_mapping[gid] = gid  # Use original ID if no symbol found
                    mapping.update(batch_mapping)
                except Exception as e:
                    print(f"Error querying MyGeneInfo: {e}")
                    continue
            # Apply mapping
            mapped_indices = [mapping.get(gid, gid) for gid in gene_ids]

        # Update the index
        expression_data.index = mapped_indices
        expression_data.index.name = 'GeneSymbol_or_ID'
        # Handle duplicates
        expression_data = handle_duplicate_indices(expression_data)

        # Update counts
        mapped_genes.update(expression_data.index)
        mapped_count += len(expression_data.index)

        # Append the mapped data to the overall data list
        all_expression_data.append(expression_data)


    else:
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
            # Identify the symbol column
            symbol_columns = ['Gene symbol', 'Symbol', 'Gene Symbol', 'SYMBOL', 'GeneSymbol']
            for col in symbol_columns:
                if col in platform_table.columns:
                    symbol_column = col
                    break
            else:
                print(f"No suitable symbol column found in platform annotation for {platform_id}.")
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

            if is_mouse:
                # Map mouse genes to human gene symbols
                platform_expression_data.index = [
                    ensembl_to_human_symbol.get(gid, mouse_symbol_to_human_symbol.get(gid, None))
                    for gid in platform_expression_data.index
                ]
                # Drop unmapped genes
                platform_expression_data = platform_expression_data[~pd.isnull(platform_expression_data.index)]

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

    # Combine data frames from all platforms or supplementary files
    combined_data = pd.concat(all_expression_data, axis=1)
    print(f"Number of genes in {gse_id} after mapping: {len(mapped_genes)}")
    print(f"First few gene symbols in {gse_id}: {list(mapped_genes)[:25]}")
    return combined_data, mapped_count, unmapped_count


def main():
    # Create input directory if it doesn't exist
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # List of GSE IDs to process
    gse_ids = ['GSE176043', 'GSE41781', 'GSE190986', 'GSE144600']  # Removed GSE148911

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
