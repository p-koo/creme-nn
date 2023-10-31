import glob
import pickle
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
import glob






def main():

    model_name = sys.argv[1]
    data_dir = '../data/'
    result_dir = f'../results/'
    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        track_index = [4824, 5110, 5111]
        model = custom_model.Enformer(track_index=track_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]
        tss_tile = [model.seq_length // 2 - 2500, model.seq_length // 2 + 2500]

    # elif model_name.lower() == 'borzoi':
    #     target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
    #     cage_rna_tracks = [i for i, t in enumerate(target_df['description']) if 'CAGE' in t or 'RNA' in t]
    #     model = custom_model.Borzoi('../data/borzoi/*/*', cage_rna_tracks, True)

    else:
        print('Unkown model')
        sys.exit(1)

    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    seq_parser = utils.SequenceParser(fasta_path)

    results_dir = utils.make_dir(f'{result_dir}/context_swap_test/')
    test_results_dir = utils.make_dir(f'{results_dir}/{model_name}/')

    dfs = {cell_line: pd.read_csv(f'../results/context_dependence_test/{model_name}/{cell_line}_context.csv') for cell_line in cell_lines}

    for cell, df in dfs.items():
        cell_line_dir = utils.make_dir(f'{test_results_dir}/{cell}')
        x_target = []
        l_target = []
        for i, row in tqdm(df.iterrows()):
            # get sequence from reference genome and convert to one-hot
            target_seq_info = row['path'].split('/')[-1].split('.')[0]
            chrom, start, strand = target_seq_info.split('_')[1:]
            l_target.append(target_seq_info)
            x_target.append(seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True))
        x_target = np.array(x_target)

        df = df.sample(frac=1)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            src_seq_info = row['path'].split('/')[-1].split('.')[0]
            result_path = f'{cell_line_dir}/{src_seq_info}.pickle'

            if not os.path.isdir(result_path):

                chrom, start, strand = src_seq_info.split('_')[1:]

                x_source = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)

                pred_mut = creme.context_swap_test(model, x_source, x_target, l_target, tss_tile)

                utils.save_pickle(result_path, pred_mut)





if __name__ == '__main__':
    main()