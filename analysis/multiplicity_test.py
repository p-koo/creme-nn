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
import copy
sys.path.append('../creme')
import creme
import custom_model
import utils
import glob


def main():
    model_name = sys.argv[1]
    track_index = int(sys.argv[2])

    perturb_window = 5000
    num_shuffle = 10
    num_iters = 15
    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir = utils.make_dir(f'../results/multiplicity_test/')
    result_dir_model = utils.make_dir(f'{result_dir}/{model_name}/')

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        bin_index = [447, 448]
        model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Optimizing for cell line:')
        print(cell_line)
    else:
        print('Unkown model')
        sys.exit(1)

    result_dir_cell = utils.make_dir(f'{result_dir_model}/{cell_line}')

    sufficient_cre_df = pd.read_csv(f'../results/sufficiency_test/{model_name}_selected_cres.csv')
    sufficient_cre_df = sufficient_cre_df[sufficient_cre_df['cell_line']==cell_line]
    sufficient_cre_df = sufficient_cre_df.sample(frac=1)

    # set up sequence parser from fasta
    seq_parser = utils.SequenceParser(fasta_path)
    tss_tile, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)

    for i, row in tqdm(sufficient_cre_df.iterrows(), total=len(sufficient_cre_df)):
        seq_id = row['seq_id']
        result_path = f"{result_dir_cell}/{seq_id}_tile_start_{row['tile_start']}_tile_end_{row['tile_end']}.pickle"
        print(result_path)
        if not os.path.isfile(result_path):
            if row['tile class'] == 'Enhancer':
                optimization = np.argmax
            elif row['tile class'] == 'Silencer':
                optimization = np.argmin
            else:
                sys.exit()

            chrom, start, strand = seq_id.split('_')[1:]
            # get seq from reference genome and convert to one-hot
            x = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)

            sufficient_tile_seq = x[row['tile_start']: row['tile_end']]
            # perform CRE Higher-order Interaction Test
            result_summary = creme.multiplicity_test(model, x, tss_tile, [row['tile_start'], row['tile_end']],
                                                     sufficient_tile_seq, cre_tiles.copy(), num_shuffle, num_iters,
                                                     optimization)

            utils.save_pickle(result_path, result_summary)


if __name__ == '__main__':
    main()
