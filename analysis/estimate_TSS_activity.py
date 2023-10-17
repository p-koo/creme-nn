import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pyranges as pr
import sys, os, h5py
import kipoiseq
from tqdm import tqdm
import json
sys.path.append('../creme')
import creme
import custom_model
import utils



def main():
    model = custom_model.Enformer()
    seq_len = 393216 # change for other models
    data_dir = '../data/'
    tracks_of_interest = [4824, 5110, 5111]
    results_dir = utils.make_dir('../results/')

    fasta_path = f'{data_dir}/hg19.fa'

    tss_csv_path = f'{results_dir}/tss_positions.csv'
    results_dir = utils.make_dir(f'{results_dir}/gencode_tss_predictions/')

    if os.path.isfile(tss_csv_path):
        tss_df = pd.read_csv(tss_csv_path, index_col=None)
    else:
        gencode_annotations = pr.read_gtf(f'{data_dir}/gencode.v43lift37.basic.annotation.gtf')
        tss_df = gencode_annotations.df.query('Feature=="transcript"')
        print(tss_df.columns)

        assert len(tss_df['Strand'].unique()) == 2, 'bad strand'
        tss_positions = [row['Start'] if row['Strand']=='+' else row['End'] for _, row in tss_df.iterrows()]
        tss_df['start'] = tss_positions
        tss_df = tss_df[['Chromosome', 'start', 'gene_name', 'gene_id', 'Strand']]
        tss_df.to_csv(tss_csv_path, index=False)
    tss_df = tss_df.sample(frac = 1)
    print(tss_df.shape)
    seq_parser = utils.SequenceParser(fasta_path)
    N = tss_df.shape[0]
    for i, row in tqdm(tss_df.iterrows(), total=N):
        assert i < N-1, 'bad index'
        result_path = f'{results_dir}/{i}.csv'
        if not os.path.isfile(result_path):
            print(f'Processing {i} to save at {result_path}')
            chrom, start = row[:2]
            sequence_one_hot = seq_parser.extract_seq_centered(chrom, start, seq_len)
            wt_pred = np.squeeze(model.predict_all(sequence_one_hot))
            pd.DataFrame(wt_pred).to_csv(result_path, index=None)


if __name__ == '__main__':
    main()
