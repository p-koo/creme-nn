import glob
import pandas as pd
import numpy as np
import sys, os
from tqdm import tqdm
import json
import copy
from creme import creme
from creme import custom_model
from creme import utils





def main():
    model_name = sys.argv[1]
    # track_index = int(sys.argv[2])

    perturb_window = 5000
    num_shuffle = 10
    num_iters = 15
    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir = utils.make_dir(f'../results/multiplicity_test/')
    result_dir_model = utils.make_dir(f'{result_dir}/{model_name}/')
    csv_dir = f'../results/summary_csvs/{model_name}/'

    print(f'USING model {model_name}')
    for track_index in [4824, 5110, 5111]:
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

        sufficient_cre_df = pd.read_csv(f'{csv_dir}/sufficient_CREs.csv')
        sufficient_cre_df = sufficient_cre_df[sufficient_cre_df['cell_line'] == cell_line]
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

                if not os.path.isfile(result_path):

                    utils.save_pickle(result_path, result_summary)
                else:
                    print('File already exists!')


    ######SUMMARIZE RESULTS
    sufficient_cre_df = pd.read_csv(f'{csv_dir}/sufficient_CREs.csv')

    multiplicity_dfs = []
    for cell, cell_df in sufficient_cre_df.groupby('cell_line'):
        for tile_type, df in cell_df.groupby('tile class'):
            for i, row in df.iterrows():
                result_path = f'{result_dir_model}/{cell}/{row["seq_id"]}_tile_start_{row["tile_start"]}_tile_end_{row["tile_end"]}.pickle'
                res = utils.read_pickle(result_path)
                res['best_tss_signal'].insert(0, res['only_tss_pred'])
                v = np.array(res['best_tss_signal']) / res['tss_and_cre_pred']  # normalize by original CRE position
                one_seq_df = pd.DataFrame(v, columns=['Normalized TSS activity'])
                one_seq_df['cell_line'] = cell
                one_seq_df['tile class'] = tile_type
                one_seq_df['seq_id'] = f"{row['seq_id']}_{row['tile_start']}"
                multiplicity_dfs.append(one_seq_df)
    multiplicity_dfs = pd.concat(multiplicity_dfs)
    multiplicity_dfs.to_csv(f"{csv_dir}/multiplicity.csv")

if __name__ == '__main__':
    main()
