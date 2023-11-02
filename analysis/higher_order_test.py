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
    track_index = int(sys.argv[2])
    optimization_name = sys.argv[3]

    if optimization_name == 'min':
        optimization = np.argmin

    elif optimization_name == 'max':
        optimization = np.argmax
    else:
        print('bad optimization')
        exit(1)

    perturb_window = 5000
    num_shuffle = 10
    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir = utils.make_dir(f'../results/higher_order_test_{optimization_name}')
    result_dir_model = utils.make_dir(f'{result_dir}/{model_name}/')

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        bin_index = [447, 448]
        model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])
        num_rounds = 25
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Optimizing for cell line:')
        print(cell_line)
    else:
        print('Unkown model')
        sys.exit(1)

    result_dir_cell = f'{result_dir_model}/{cell_line}'

    context_df = pd.read_csv(f'../results/context_dependence_test/{model_name}/{cell_line}_context.csv')

    context_df = context_df.sample(frac=1)

    # get coordinates of central tss
    _, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)
    # set up sequence parser from fasta
    seq_parser = utils.SequenceParser(fasta_path)

    for i, row in tqdm(context_df.iterrows(), total=len(context_df)):
        seq_id = row['path'].split('/')[-1].split('.')[0]
        result_path = f'{result_dir_cell}/{seq_id}.pickle'
        print(result_path)
        if not os.path.isfile(result_path):
            chrom, start, strand = seq_id.split('_')[1:]
            # get seq from reference genome and convert to one-hot
            x = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)

            # perform CRE Higher-order Interaction Test
            result_summary = creme.higher_order_interaction_test(model, x, cre_tiles, optimization, num_shuffle,
                                                                 num_rounds)

            utils.save_pickle(result_path, result_summary)


if __name__ == '__main__':
    main()
