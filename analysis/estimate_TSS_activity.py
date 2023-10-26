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

    model_name = sys.argv[1]
    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        # target_df = pd.read_csv('../data/enformer_targets_human.txt', sep='\t')
        # cage_tracks = [i for i, t in enumerate(target_df['description']) if 'CAGE' in t]
        model = custom_model.Enformer()
    elif model_name.lower() == 'borzoi':
        target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
        cage_rna_tracks = [i for i, t in enumerate(target_df['description']) if 'CAGE' in t or 'RNA' in t]
        target_df.iloc[cage_rna_tracks].to_csv('../data/borzoi_cage_rna_tracks.csv')
        model = custom_model.Borzoi('../data/borzoi/*/*', cage_rna_tracks, True, [8174, 8175, 8176, 8177])

    else:
        print('Unkown model')
        sys.exit(1)

    seq_len = model.seq_length # change for other models
    data_dir = '../data/'
    results_dir = utils.make_dir('../results/')

    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'

    tss_csv_path = f'{results_dir}/tss_positions.csv'
    results_dir = utils.make_dir(f'{results_dir}/gencode_tss_predictions/')
    results_dir = utils.make_dir(f'{results_dir}/{model_name}/')

    if os.path.isfile(tss_csv_path):
        tss_df = pd.read_csv(tss_csv_path, index_col=None)
    else:
        gencode_annotations = pr.read_gtf(f'{data_dir}/gencode.v44.basic.annotation.gtf')
        tss_df = gencode_annotations.df.query('Feature=="transcript" & gene_type == "protein_coding"')
        print(tss_df.shape)

        assert len(tss_df['Strand'].unique()) == 2, 'bad strand'
        tss_positions = [row['Start'] if row['Strand']=='+' else row['End'] for _, row in tss_df.iterrows()]
        tss_df['start'] = tss_positions
        tss_df = tss_df[['Chromosome', 'start', 'gene_name', 'gene_id', 'Strand']]
        tss_df.to_csv(tss_csv_path, index=False)
    tss_df = tss_df.sample(frac = 1)

    seq_parser = utils.SequenceParser(fasta_path)
    N = tss_df.shape[0]
    print(N)
    for j, (i, row) in tqdm(enumerate(tss_df.iterrows()), total=N):
        result_path = f'{results_dir}/{i}.npy'
        print(result_path)
        assert j < N, 'bad index'
        if not os.path.isfile(result_path):
            chrom, start = row[:2]
            strand = row['Strand']
            sequence_one_hot = seq_parser.extract_seq_centered(chrom, start, strand, seq_len)
            wt_pred = np.squeeze(model.predict(sequence_one_hot))
            np.save(result_path, wt_pred)


if __name__ == '__main__':
    main()
