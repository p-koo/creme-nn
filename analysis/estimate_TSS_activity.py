import glob

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
import gene as bgene


def main():

    model_name = sys.argv[1]
    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        # target_df = pd.read_csv('../data/enformer_targets_human.txt', sep='\t')
        # cage_tracks = [i for i, t in enumerate(target_df['description']) if 'CAGE' in t]
        model = custom_model.Enformer()
    elif model_name.lower() == 'borzoi':
        target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
        cage_tracks = [i for i, t in enumerate(target_df['description']) if
                       ('CAGE' in t) and (t.split(':')[-1].strip() in ['K562 ENCODE, biol_',
                                                                       'GM12878 ENCODE, biol_',
                                                                       'PC-3'])]

        rna_tracks = [i for i, t in enumerate(target_df['description']) if
                      ('RNA' in t) and (t.split(':')[-1].strip() in ['K562',
                                                                     'GM12878',
                                                                     'PC-3'])]
        target_df.iloc[cage_tracks+rna_tracks].to_csv('../data/borzoi_cage_rna_tracks.csv')
        model = custom_model.Borzoi('../data/borzoi/*/*', cage_tracks, rna_tracks, aggregate=True)

    else:
        print('Unkown model')
        sys.exit(1)

    seq_len = model.seq_length # change for other models
    data_dir = '../data/'
    results_dir = utils.make_dir('../results/')
    transcriptome = bgene.Transcriptome('../data/gencode.v44.basic.annotation.gtf')
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'

    tss_csv_path = f'{results_dir}/tss_positions.csv'
    results_dir = utils.make_dir(f'{results_dir}/gencode_tss_predictions/')
    results_dir = utils.make_dir(f'{results_dir}/{model_name}/')

    if os.path.isfile(tss_csv_path):
        tss_df = pd.read_csv(tss_csv_path, index_col=None)
    else:
        gencode_annotations = pr.read_gtf(f'{data_dir}/gencode.v44.basic.annotation.gtf')
        tss_df = gencode_annotations.df.query('Feature=="transcript" & gene_type == "protein_coding"')
        tss_df = tss_df.drop_duplicates(subset=['Chromosome', 'Start', 'Strand'])
        print(tss_df.shape)

        assert len(tss_df['Strand'].unique()) == 2, 'bad strand'
        tss_positions = [row['Start'] if row['Strand']=='+' else row['End'] for _, row in tss_df.iterrows()]
        tss_df['Start'] = tss_positions
        tss_df = tss_df[['Chromosome', 'Start', 'gene_name', 'gene_id', 'Strand']]
        tss_df.to_csv(tss_csv_path, index=False)
    tss_df = tss_df.sample(frac = 1)

    seq_parser = utils.SequenceParser(fasta_path)
    N = tss_df.shape[0]
    print(N)
    for j, (i, row) in tqdm(enumerate(tss_df.iterrows()), total=N):
        chrom, start = row[:2]
        strand = row['Strand']
        result_path_prefix = f'{results_dir}/{utils.get_summary(row)}'
        print(result_path_prefix)
        assert j < N, 'bad index'
        if len(glob.glob(f'{result_path_prefix}*')) == 0: # if result does not exist

            sequence_one_hot = seq_parser.extract_seq_centered(chrom, start, strand, seq_len)
            if model_name == 'enformer':
                wt_pred = np.squeeze(model.predict(sequence_one_hot))
                np.save(f'{result_path_prefix}.npy', wt_pred)
            elif model_name == 'borzoi':
                cage_bins = np.arange(model.target_lengths // 2-4, model.target_lengths // 2+4, 1)
                gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if row['gene_id'] in gene_key]
                gene = transcriptome.genes[gene_keys[0]]
                wt_pred = model.predict_cage_rna(sequence_one_hot, gene, start, cage_bins=cage_bins, return_exons=True,
                                                 return_all=False)

                utils.save_pickle(f'{result_path_prefix}.pickle', wt_pred)



if __name__ == '__main__':
    main()
