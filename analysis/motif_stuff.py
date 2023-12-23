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



shuffle_num = 10
minitile_size = 50
enhancer_definition = 0.3
threshold = 0.75
N_batch = 5


model_name = 'enformer'
seq_parser = utils.SequenceParser('../data/GRCh38.primary_assembly.genome.fa')
csv_dir = f'../results/summary_csvs/{model_name}/'
suff_cre_df = pd.read_csv(f'{csv_dir}/sufficient_CREs.csv')
cre_df_all_cells = suff_cre_df[suff_cre_df['tile class']=='Enhancer']

tile_coords = pd.read_csv(f'{csv_dir}/sufficiency_test_tile_coordinates.csv', index_col='Unnamed: 0').T
tss_tile = tile_coords.loc['tss'].T.values
cre_tile_coords = tile_coords.loc[[t for t in tile_coords.index if 'tss' not in t]]
tile_size = cre_tile_coords[0][1] - cre_tile_coords[0][0]


track_indeces = [4824, 5110, 5111]
data_dir = '../data/'
target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_indeces]

bin_index = [447, 448]

minitile_dir = utils.make_dir(f'../results/motifs_{minitile_size}_batch_{N_batch}_shuffle_{shuffle_num}_thresh_{threshold}')





for track_index, cell_line in zip(track_indeces, cell_lines):
    print('!!!!!!!!!')
    print(track_index, cell_line)
    model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)

    cre_df = cre_df_all_cells[cre_df_all_cells['cell_line']==cell_line]
    cre_df = cre_df.sample(frac=1)

    outdir = utils.make_dir(f'{minitile_dir}/{cell_line}')

    for i, row in tqdm(cre_df.iterrows(), total=cre_df.shape[0]):
        result_path = f"{outdir}/{row['seq_id']}.pickle"
        print(result_path)
        if not os.path.isfile(result_path):
            per_seq_results = []
            chrom, tss_site, strand = row['seq_id'].split('_')[1:]
            wt_seq = seq_parser.extract_seq_centered(chrom, int(tss_site), strand, model.seq_length)  # get seq

            pred_wt, pred_mut, pred_control, control_sequences = creme.sufficiency_test(model, wt_seq, tss_tile,
                                                                                        [[row['tile_start'],
                                                                                          row['tile_end']]],
                                                                                        shuffle_num, mean=False,
                                                                                        return_seqs=True)
            wt = pred_wt.mean()
            mut = pred_mut.mean()
            control = pred_control.mean()
            result_summary = {'wt': wt, 'mut': mut, 'control': control}
            if (mut - control) / wt > enhancer_definition:

                minitile_starts = list(
                    range(row['tile_start'], row['tile_end'] - minitile_size + 1, minitile_size // 2))
                number_of_tiles = len(minitile_starts)
                score = 1  # start with no minitile removed
                removed_tiles = np.array([])  # none selected at start
                new_selected_tiles = []
                remaining_to_test = minitile_starts.copy()  # start with full list of minitile starts
                pruned_seqs = control_sequences.copy()  # 10 shuffled versions with just TSS
                pruned_seqs[:, row['tile_start']:row['tile_end'], :] = (
                wt_seq[row['tile_start']:row['tile_end']]).copy()  # start with intact CRE
                while score > threshold:
                    print(score)
                    results = []  # [len = number of remaining tiles to test]

                    # remove one minitile at a time
                    for j, minitile_start in tqdm(enumerate(remaining_to_test), total=len(remaining_to_test)):
                        minitile_end = minitile_start + minitile_size
                        # "remove" minitile by putting back shuffled version
                        seq_extra_minitile_shuffled = pruned_seqs.copy()
                        # use the control sequence to shuffle minitile
                        seq_extra_minitile_shuffled[:, minitile_start: minitile_end, :] = control_sequences[:,
                                                                                          minitile_start: minitile_end,
                                                                                          :].copy()
                        results.append(model.predict(seq_extra_minitile_shuffled).mean())

                    new_selected_tiles = np.array(remaining_to_test)[np.argsort(results)[-N_batch:]]  # choose N useless
                    removed_tiles = np.concatenate([removed_tiles, new_selected_tiles])  # add to santa's bad list

                    # prune the list and sequences to test based on this iteration
                    for removed_tile in new_selected_tiles:
                        remaining_to_test.remove(removed_tile)
                        pruned_seqs[:, removed_tile:removed_tile + minitile_size, :] = (
                        control_sequences[:, removed_tile:removed_tile + minitile_size, :]).copy()

                    # compute new score
                    score = model.predict(pruned_seqs).mean() / mut
                    per_seq_results.append(score)
                result_summary['fraction_explained'] = per_seq_results
            utils.save_pickle(result_path, result_summary)