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
    perturb_window = 5000
    num_shuffle = 10
    data_dir = '../data/'
    csv_dir = f'../results/summary_csvs/{model_name}/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir_model = utils.make_dir(f"{utils.make_dir(f'../results/necessity_test')}/{model_name}/")

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        track_index = [4824, 5110, 5111]
        bin_index = [447, 448]
        model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]

    else:
        print('Unkown model')
        sys.exit(1)

    context_dfs_per_cell = {cell_line: pd.read_csv(f'{csv_dir}/{cell_line}_selected_contexts.csv')
                            for cell_line in cell_lines}
    
    context_df = pd.concat(context_dfs_per_cell.values()).drop_duplicates('path')

    context_df = context_df.sample(frac = 1)


    # get coordinates of central tss
    tss_tile, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)
    tile_df = pd.DataFrame(cre_tiles).T
    tile_df['tss'] = tss_tile
    tile_df.to_csv(f'{result_dir_model}/tile_coordinates.csv')
    # set up sequence parser from fasta
    seq_parser = utils.SequenceParser(fasta_path)


    for i, row in tqdm(context_df.iterrows(), total=len(context_df)):
        seq_id = row['path'].split('/')[-1].split('.')[0]
        result_path = f'{result_dir_model}/{seq_id}.pickle'
        print(result_path)
        if not os.path.isfile(result_path):
            chrom, start, strand = seq_id.split('_')[1:]
            # get seq from reference genome and convert to one-hot
            x = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)

            # perform CRE Necessity Test
            pred_wt, pred_mut, std_mut = creme.necessity_test(model, x, cre_tiles, num_shuffle, mean=True)
            utils.save_pickle(result_path, {'wt': pred_wt, 'mut': pred_mut, 'mut_std': std_mut})


    ######### SUMMARIZE RESULTS
    cre_tile_coords = pd.DataFrame(cre_tiles)
    result_summary = []
    for c, cell_line in enumerate(cell_lines):
        cell_line_context = context_dfs_per_cell[cell_line]
        print(c, cell_line)
        for _, row in cell_line_context.iterrows():
            res_path = f'{result_dir_model}/{row["seq_id"]}.pickle'
            res_raw = utils.read_pickle(res_path)
            res = {k: r[:, :, c].mean(axis=1) for k, r in res_raw.items()}
            # res['mut'] = np.delete(res['mut'], 19)
            one_seq = pd.DataFrame((res['wt'] - res['mut']) / res['wt'])
            one_seq.columns = ['Normalized shuffle effect']
            one_seq['seq_id'] = row["seq_id"]
            one_seq['tile_start'] = cre_tile_coords[0].values
            one_seq['tile_end'] = cre_tile_coords[1].values
            one_seq['context'] = row['context']
            one_seq['cell_line'] = cell_line
            result_summary.append(one_seq)
    result_summary = pd.concat(result_summary)

    result_summary.to_csv(f'{csv_dir}/necessity_test.csv')


    ######## DEFINE ENHANCING AND SILENCING CRES BASED ON NECESSITY

    selected_cres = []
    for cell, df in result_summary.groupby('cell_line'):
        df['cell_line'] = cell
        enh_df = (df[(df['Normalized shuffle effect'] > 0.3) & (df['context'] == 'enhancing')]).copy()
        enh_df['tile class'] = 'Enhancer'
        selected_cres.append(enh_df)
        sil_df = (df[(df['Normalized shuffle effect'] < -0.3) & (df['context'] == 'silencing')]).copy()
        sil_df['tile class'] = 'Silencer'
        selected_cres.append(sil_df)
    selected_cres = pd.concat(selected_cres)
    selected_cres.to_csv(f'{csv_dir}/necessary_CREs.csv')

if __name__ == '__main__':
    main()



