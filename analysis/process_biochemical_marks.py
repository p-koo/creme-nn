import pandas as pd
import pyBigWig

import numpy as np
import sys
sys.path.append('../creme/')
import utils


def main():
    res_dir = utils.make_dir('../results/biochemical_marks')
    cre_classifier = 'sufficiency'
    cre_size = 5000
    cres = pd.read_csv(f'../results/summary_csvs/enformer/sufficient_CREs.csv')

    model_seq_length = 196608
    for assay in ['histone', 'accessibility', 'tf']:
        all_features = []
        for cell_line in ['K562', 'GM12878', 'PC-3']:
            assay_dir = f'../data/biochemical_marks/{cell_line}/{assay}'
            metadata = pd.read_csv(f'{assay_dir}/metadata.csv')
            cell_df = cres[cres['cell_line'] == cell_line]

            for m, metadata_row in metadata.iterrows():  # per bigwig track
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
                        chrom, tss = row['seq_id'].split('_')[1:3]
                        seq_start = int(tss) - model_seq_length // 2
                        cre_start = seq_start + row['tile_start']
                        cre_end = cre_start + cre_size
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