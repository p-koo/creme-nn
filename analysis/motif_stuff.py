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


def main():
    scales = [int(i) for i in sys.argv[1].split(',')]
    thresholds = [float(i) for i in sys.argv[2].split(',')]
    N_batch = int(sys.argv[3])

    print("Optimizing for: ")
    for s, t in zip(scales, thresholds):
        print(f'{s} at threshold {t}')

    shuffle_num = 10
    frac = 1
    enhancer_definition = 0.3



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

    minitile_dir = utils.make_dir(f'../results/motifs_{sys.argv[1]}_batch_{N_batch}_shuffle_{shuffle_num}_thresh_{sys.argv[2]}')
    print(f'Results will be saved in {minitile_dir}')




    for track_index, cell_line in zip(track_indeces, cell_lines):
        print('!!!!!!!!!')
        print(track_index, cell_line)
        model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)

        cre_df = cre_df_all_cells[cre_df_all_cells['cell_line']==cell_line]
        cre_df = cre_df.sample(frac=1)

        outdir = utils.make_dir(f'{minitile_dir}/{cell_line}')

        for i, row in tqdm(cre_df.iterrows(), total=cre_df.shape[0]):
            result_path = f"{outdir}/{row['seq_id']}_{row['tile_start']}_{row['tile_end']}.pickle"
            print(result_path)
            if not os.path.isfile(result_path):
                per_seq_results = []
                chrom, tss_site, strand = row['seq_id'].split('_')[1:]
                wt_seq = seq_parser.extract_seq_centered(chrom, int(tss_site), strand, model.seq_length)  # get seq
                whole_tile_start, whole_tile_end = [row['tile_start'], row['tile_end']]
                pred_wt, pred_mut, pred_control, control_sequences = creme.sufficiency_test(model, wt_seq, tss_tile,
                                                                                            [[whole_tile_start,
                                                                                              whole_tile_end]],
                                                                                            shuffle_num, mean=False,
                                                                                            return_seqs=True)
                wt = pred_wt.mean()
                mut = pred_mut.mean()
                control = pred_control.mean()
                min_results = {'wt': wt, 'mut': mut, 'control': control}
                if (mut - control) / wt > enhancer_definition:

                    opt_results = creme.prune_sequence(model, wt_seq, control_sequences, mut, whole_tile_start, whole_tile_end,
                                         scales, thresholds, frac, N_batch)

                    result_summary = min_results.update(opt_results)
                if not os.path.isfile(result_path):

                    utils.save_pickle(result_path, result_summary)
                else:
                    print('File already exists!')

if __name__ == "__main__":
    main()