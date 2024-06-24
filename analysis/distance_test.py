import glob
import pandas as pd
import seaborn as sns
import numpy as np
import sys, os
from tqdm import tqdm
import json

sys.path.append('./borzoi')
import borzoi_custom_model

from creme import creme
from creme import custom_model
from creme import utils


########################################################################################
# parameters
########################################################################################

def main():

    if len(sys.argv) == 4:
        model_name = sys.argv[1]
        num_shuffle = int(sys.argv[2])
        set_seed = bool(sys.argv[3])
    else:
        model_name = sys.argv[1]
        num_shuffle = int(sys.argv[2])
        set_seed = False

    perturb_window = 5000

    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir = utils.make_dir(f'../results/distance_test_{set_seed}')
    result_dir_model = utils.make_dir(f'{result_dir}/{model_name}_{num_shuffle}/')
    csv_dir = f'../results/summary_csvs/{model_name}/'
    print(f'USING model {model_name}')
    cre_df_path = f'{csv_dir}/sufficient_CREs.csv'

    if model_name.lower() == 'enformer':
        track_index = [4824, 5110, 5111]
        model = custom_model.Enformer(track_index=track_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]
        compute_mean = True
    elif model_name.lower() == 'borzoi':
        target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
        cell_lines_for_search = ['K562 ENCODE, biol_']
        cell_line_info, cage_tracks = utils.get_borzoi_targets(target_df, cell_lines_for_search)
        print('Loading Borzoi(s)')
        model = borzoi_custom_model.Borzoi('../data/borzoi/*/*', track_index=cage_tracks, aggregate=True)
        model.bin_index = list(np.arange(model.target_lengths // 2 - 4, model.target_lengths // 2 + 4, 1))
        compute_mean = False
    else:
        print('Unkown model')
        sys.exit(1)


    # load CRE dataframe (with all CRE positions)
    cre_df = pd.read_csv(cre_df_path)



    tss_tile, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)
    cre_tiles_starts = np.array(cre_tiles)[:, 0]
    cre_tiles_starts_abs = (cre_tiles_starts - tss_tile[0]) // 1000
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

            res = creme.distance_test(model, x, tss_tile, [tile_start, tile_end],
                                       cre_tiles_starts, num_shuffle, mean=compute_mean,
                                       seed=set_seed)

            # store predictions
            utils.save_pickle(result_path, res)

    if model_name == 'enformer':
        result_normalized_effects = []
        all_norm_effects = []
        raw_preds = {}
        for i, cell_line in enumerate(cell_lines):
            raw_preds[cell_line] = []
            cre_df_cell = cre_df[cre_df['cell_line'] == cell_line]
            cre_df_cell.insert(1, "distance to TSS (Kb)",
                               [(int(i) - tss_tile[0]) // 1000 for i in cre_df_cell['tile_start'].values])
            for j, (_, row) in tqdm(enumerate(cre_df_cell.iterrows())):
                tile_start, tile_end = [row['tile_start'], row['tile_end']]
                result_path = f'{result_dir_model}/{row["seq_id"]}_{tile_start}_{tile_end}.pickle'
                res = utils.read_pickle(result_path)
                control = row['control']  # res['mean_control'][447:449,i].mean()
                test = res['mean_mut'][:, 447:449, i].mean(axis=-1)
                if row['context'] == 'enhancing':
                    CRE_norm_effects = (test - control) / row['wt']
                else:
                    CRE_norm_effects = (test - control) / control

                norm_effects = test / np.max(test)
                all_norm_effects.append(norm_effects)
                df = pd.DataFrame([norm_effects, CRE_norm_effects, cre_tiles_starts_abs]).T
                df.columns = ['Fold change over control', "CRE sufficiency effect", 'Binned distance (Kb)']
                df['Normalized CRE effect (control)'] = row['Normalized CRE effect']
                raw_preds[cell_line].append(test)
                df['cell line'] = cell_line
                df['control'] = control
                df['enf_data_id'] = f'{row["seq_id"]}_{tile_start}_{tile_end}'
                df['context'] = row['context']
                df['tile class'] = row['tile class']
                result_normalized_effects.append(df)

        result_normalized_effects = pd.concat(result_normalized_effects)
        result_normalized_effects.to_csv(f"{csv_dir}/distance_test.csv")

if __name__ == '__main__':
    main()


