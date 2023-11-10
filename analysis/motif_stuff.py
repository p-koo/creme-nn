import pandas as pd
import numpy as np
import sys
sys.path.append('../creme/')
import custom_model
import creme
import utils
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pyranges as pr
import shutil
import pickle
import kipoiseq
import os


track_index = [5111]
bin_index = [447, 448]
data_dir = '../data/'
model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]
model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
cell_channel = np.argwhere(np.array(cell_lines) == 'K562').flatten()[0]
model_name = 'enformer'

shuffle_num = 5

seq_parser = utils.SequenceParser('../data/GRCh38.primary_assembly.genome.fa')

cre_df = pd.read_csv(f'../results/necessity_test/{model_name}_selected_cres.csv')
cre_df = cre_df[cre_df['tile class']=='Enhancer']
cre_df = cre_df[cre_df['cell_line']=='K562']

tile_coords = pd.read_csv(f'../results/sufficiency_test/{model_name}/tile_coordinates.csv', index_col='Unnamed: 0').T
tss_tile = tile_coords.loc['tss'].T.values
cre_tile_coords = tile_coords.loc[[t for t in tile_coords.index if 'tss' not in t]]

tile_size = cre_tile_coords[0][1] - cre_tile_coords[0][0]

minitile_size = 500
step_size = minitile_size
cre_df = cre_df.sample(frac=1)
minitile_effects = {}
tile_effects = {}
outdir = utils.make_dir('../results/motifs/')

for row_i, (_, row) in enumerate(cre_df.iterrows()):
    result_path = f'{outdir}/{row["seq_id"]}'
    print(result_path)
    if not os.path.isfile(result_path):

        minitile_effects[row['seq_id']] = []

        for minitile_size in [25, 50, 100, 200, 250, 500, 1000]:

            chrom, tss_site, strand = row['seq_id'].split('_')[1:]
            wt_seq = seq_parser.extract_seq_centered(chrom, int(tss_site), strand, model.seq_length)
            tile_index = np.argwhere(np.array(cre_tile_coords) == [row['tile_start'], row['tile_end']])[0][0]
            pred_wt, pred_mut, std_mut, tile_mut_seqs = creme.necessity_test(model, wt_seq,
                                                                             [[row['tile_start'], row['tile_end']]],
                                                                             shuffle_num, mean=True, return_seqs=True)
            minitile_starts = list(range(row['tile_start'], row['tile_end'] - minitile_size + 1, minitile_size))
            number_of_tiles = len(minitile_starts)
            minitile_add_res = np.empty((shuffle_num, number_of_tiles, len(bin_index), len(track_index)))

            for i, tile_removed_seq in enumerate(np.squeeze(tile_mut_seqs)):
                for j, minitile_start in tqdm(enumerate(minitile_starts), total=number_of_tiles):
                    x = tile_removed_seq.copy()
                    minitile_seq = wt_seq[minitile_start:minitile_start + minitile_size].copy()
                    x[minitile_start:minitile_start + minitile_size] = minitile_seq
                    mut_plus_minitile_pred = model.predict(x)[0]
                    minitile_add_res[i, j, ...] = mut_plus_minitile_pred
            normalized_mini_effects = minitile_add_res.min(axis=0)[:, :, 0].mean(axis=-1) / pred_wt[0, :,
                                                                                            cell_channel].mean()
            normalized_mini_effects = np.repeat(np.array(normalized_mini_effects).flatten(), minitile_size // 25)
            minitile_effects[row['seq_id']].append(normalized_mini_effects)
            minitile_effects[f"{row['seq_id']}_control"] = row['Normalized shuffle effect']
        utils.save_pickle(result_path, minitile_effects)
