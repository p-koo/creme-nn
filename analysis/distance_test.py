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


########################################################################################
# parameters
########################################################################################

def main():

    model_name = sys.argv[1]
    perturb_window = 5000
    num_shuffle = 10
    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir = utils.make_dir(f'../results/distance_test')
    result_dir_model = utils.make_dir(f'{result_dir}/{model_name}/')
    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        track_index = [4824, 5110, 5111]
        model = custom_model.Enformer(track_index=track_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]


    else:
        print('Unkown model')
        sys.exit(1)


    # load CRE dataframe (with all CRE positions)
    cre_df = pd.read_csv(f'../results/sufficiency_test/{model_name}_selected_cres.csv')
    tile_coords = pd.read_csv(f'../results/sufficiency_test/{model_name}/tile_coordinates.csv', index_col='Unnamed: 0').T
    tss_tile = tile_coords.loc['tss'].T.values
    cre_tile_coords = tile_coords.loc[[t for t in tile_coords.index if 'tss' not in t]]
    cre_tiles_starts = cre_tile_coords[0].values

    # set up sequence parser from fasta
    seq_parser = utils.SequenceParser(fasta_path)
    cre_df = cre_df.sample(frac=1)
    # loop through and predict TSS activity
    for i, row in tqdm(cre_df.iterrows(), total=len(cre_df)):
        tile_start, tile_end = [row['tile_start'], row['tile_end']]
        result_path = f'{result_dir_model}/{row["seq_id"]}_{tile_start}_{tile_end}.pickle'
        print(result_path)
        if not os.path.isfile(result_path):
            # get sequence from reference genome and convert to one-hot
            chrom, start, strand = row['seq_id'].split('_')[1:]
            x = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)

            # perform TSS-CRE distance dependence Test
            mean_control, std_control, mean_mut, std_mut = creme.distance_test(model, x, tss_tile, [tile_start, tile_end],
                                                                               cre_tiles_starts, num_shuffle, mean=True)

            # store predictions
            utils.save_pickle(result_path, {"mean_control": mean_control, "std_control": std_control,
                                            "mean_mut": mean_mut, "std_mut": std_mut})

if __name__ == '__main__':
    main()


