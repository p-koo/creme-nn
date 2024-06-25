import pandas as pd
import pyBigWig
from tqdm import tqdm
import numpy as np
import sys
from creme import utils



def main():
    res_dir = utils.make_dir('../results/biochemical_marks')
    cre_size = 5000
    # load enhancing and silencing tiles
    selected_cres = pd.read_csv(f'../results/summary_csvs/enformer/sufficient_CREs.csv')

    # Load neutral tiles
    suff_test_res = pd.read_csv('../results/summary_csvs/enformer/sufficiency_test.csv')
    neutral_tiles = suff_test_res[
        (suff_test_res['context'] == 'neutral') & (np.abs(suff_test_res['(MUT - CONTROL) / CONTROL']) < 0.1)]
    unique_genes = []
    for _, df in neutral_tiles.sample(frac=1).groupby('seq_id'):
        unique_genes.append(df.iloc[0])
    neutrals = pd.DataFrame(unique_genes)
    neutrals['tile class'] = 'Neutral'
    neutrals['Normalized CRE effect'] = neutrals['(MUT - CONTROL) / CONTROL']
    cres = pd.concat([selected_cres, neutrals])

    #### Subsample neutrals
    print('Sampling neutral tiles')
    subsampled = []
    for cell_line, df in cres.groupby('cell_line'):
        df.groupby('tile class').count()
        N_per_class = df.groupby('tile class').count().iloc[:, 0].to_dict()
        target_N = (N_per_class['Silencer'] + N_per_class['Enhancer']) // 2
        for tile_class, tile_df in df.groupby('tile class'):
            if tile_class == 'Neutral':
                tile_df = tile_df.iloc[:target_N, :]
            subsampled.append(tile_df)
    cres = pd.concat(subsampled)

    model_seq_length = 196608
    for assay in ['histone', 'accessibility', 'tf']:
        print(f'Processing {assay} marks')
        all_features = []
        for cell_line in ['K562', 'GM12878', 'PC-3']:
            print(f'Processing {cell_line}')
            assay_dir = f'../data/biochemical_marks/{cell_line}/{assay}'
            metadata = pd.read_csv(f'{assay_dir}/metadata.csv')
            cell_df = cres[cres['cell_line'] == cell_line]

            for m, metadata_row in tqdm(metadata.iterrows()):  # per bigwig track
                # get generic descriptors
                bw_id = metadata_row['File accession']

                if assay == 'accessibility':
                    descr = metadata_row['Assay']
                else:
                    descr = metadata_row['Experiment target'].split('-')[0]

                # read in values
                bw = pyBigWig.open(f'{assay_dir}/{bw_id}.bigWig')

                for cre_type, df in cell_df.groupby('tile class'):
                    vals_mean = []
                    vals_max = []
                    cre_types = []
                    cre_id = []
                    for i, row in df.iterrows():
                        chrom, tss, strand = row['seq_id'].split('_')[1:]
                        tss = int(tss)
                        cre_midpoint = row['tile_end'] - cre_size // 2
                        delta = cre_midpoint - model_seq_length // 2
                        if strand == "+":
                            cre_midpoint_coord = tss + delta
                        elif strand == '-':
                            cre_midpoint_coord = tss - delta
                        cre_start = cre_midpoint_coord - cre_size // 2
                        cre_end = cre_midpoint_coord + cre_size // 2
                        v = bw.values(chrom, cre_start, cre_end)
                        vals_max.append(np.max(v))
                        vals_mean.append(np.mean(v))
                        cre_types.append(cre_type)
                        cre_id.append(f"{chrom}_{cre_start}_{cre_end}")
                    df = pd.DataFrame([vals_mean, vals_max, cre_types, cre_id]).T
                    df.columns = ['Mean coverage', 'Max coverage', 'CRE type', 'CRE id']
                    df['assay'] = assay
                    df['cell line'] = cell_line
                    df['epigenetic mark'] = descr
                    df['track_id'] = m

                    all_features.append(df)
                bw.close()

        all_features = pd.concat(all_features)

        all_features.to_csv(f'{res_dir}/{assay}.csv')


if __name__ == '__main__':
    main()