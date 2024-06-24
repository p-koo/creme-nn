import glob
import pickle
import pandas as pd
import numpy as np
import sys, os
from tqdm import tqdm
import json


sys.path.append('./borzoi')
import borzoi_custom_model

from creme import creme
from creme import custom_model
from creme import utils






def main():

    model_name = sys.argv[1]
    num_shuffle = int(sys.argv[2])
    perturb_window = 5000

    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    result_dir_model = utils.make_dir(f"{utils.make_dir(f'../results/sufficiency_test')}/{model_name}/")
    csv_dir = f'../results/summary_csvs/{model_name}/'

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        track_index = [4824, 5110, 5111]
        bin_index = [447, 448]

        model = custom_model.Enformer(track_index=track_index)
        target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
        cell_lines = [utils.clean_cell_name(target_df.iloc[t]['description']) for t in track_index]

        context_dfs_per_cell = {cell_line: pd.read_csv(f'{csv_dir}/{cell_line}_selected_contexts.csv')
                                for cell_line in cell_lines}
        context_df = pd.concat(context_dfs_per_cell.values()).drop_duplicates('path')

    elif model_name == 'borzoi':
        target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
        cell_lines_for_search = ['K562 ENCODE, biol_', 'GM12878 ENCODE, biol_', 'PC-3']
        track_index = [i for i, t in enumerate(target_df['description']) if
                       ('CAGE' in t) and (t.split(':')[-1].strip() in cell_lines_for_search)]
        cell_line_info = {}
        for target_cell_line in cell_lines_for_search:
            cell_line_info[target_cell_line] = {}
            targets = [i for i, t in enumerate(target_df['description']) if
                       ('CAGE' in t) and (t.split(':')[-1].strip() == target_cell_line)]

            cell_line_info[target_cell_line]['output'] = [np.argwhere(np.array(track_index) == t).flatten()[0] for t in
                                                          targets]
            cell_line_info[target_cell_line]['target'] = '&'.join([str(t) for t in targets])
        print('Loading Borzoi(s)')
        model = custom_model.Borzoi('../data/borzoi/*/*', track_index=track_index, aggregate=True)
        model.bin_index = list(np.arange(model.target_lengths // 2 - 4, model.target_lengths // 2 + 4, 1))

        context_dfs_per_cell = {cell_line.split()[0]: pd.read_csv(f'{csv_dir}/{cell_line.split()[0]}_selected_contexts.csv')
                                for cell_line in cell_lines_for_search}
        context_df = pd.concat(context_dfs_per_cell.values()).drop_duplicates('path')
        print(context_df.shape)

    else:
        print('Unkown model')
        sys.exit(1)


    

    context_df = context_df.sample(frac=1)
    # get coordinates of central tss
    tss_tile, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)
    tile_df = pd.DataFrame(cre_tiles).T
    tile_df['tss'] = tss_tile
    tile_df.to_csv(f'{csv_dir}/sufficiency_test_tile_coordinates.csv')
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
            pred_wt, pred_mut_mean, pred_mut_std, pred_control_mean, pred_control_std = creme.sufficiency_test(model, x,
                                                                                                               tss_tile,
                                                                                                             cre_tiles,
                                                                                                             num_shuffle,
                                                                                                             mean=True)
            result_dict = {'wt': pred_wt, 'mut': pred_mut_mean, 'mut_std': pred_mut_std,
                                            'control': pred_control_mean, 'control_std': pred_control_std}


            if not os.path.isfile(result_path):
                utils.save_pickle(result_path, result_dict)

    if model_name == 'enformer':
        ######## SUMMARIZE RESULTS

        tile_df = pd.DataFrame(cre_tiles)
        result_summary = []
        for c, cell_line in enumerate(cell_lines):
            cell_line_context = context_dfs_per_cell[cell_line]
            print(c, cell_line)
            for _, row in cell_line_context.iterrows():
                res_path = f'{result_dir_model}/{row["seq_id"]}.pickle'
                res = utils.read_pickle(res_path)
                res['wt'] = res['wt'][bin_index, c].mean(axis=0)
                res['mut'] = res['mut'][:, bin_index, c].mean(axis=1)
                res['control'] = res['control'][:, bin_index, c].mean(axis=1)
                # one_seq = pd.DataFrame((res['mut'] - res['control']) / res['wt'])
                one_seq = pd.DataFrame((res['mut']-res['control']) / res['wt'])

                one_seq.columns = ['(MUT - CONTROL) / WT']
                one_seq['(MUT - CONTROL) / CONTROL'] = (res['mut']-res['control']) / res['control']
                one_seq['seq_id'] = row['seq_id']
                one_seq['control'] = res['control']
                one_seq['wt'] = res['wt']
                one_seq['mut'] = res['mut']
                one_seq['tile_start'] = tile_df[0].values
                one_seq['tile_end'] = tile_df[1].values
                one_seq['context'] = row['context']
                one_seq['cell_line'] = cell_line
                result_summary.append(one_seq)
        result_summary = pd.concat(result_summary)
        result_summary.to_csv(f'{csv_dir}/sufficiency_test.csv')

        ########### SELECT SUFFICIENT CRES

        selected_cres = []
        for cell, df in result_summary.groupby('cell_line'):
            enh_cont_df = (df[df['context'] == 'enhancing']).copy()  # only select enhancing CREs in enhancing contexts
            sil_cont_df = (df[df['context'] == 'silencing']).copy()  # only select silencing CREs in silencing contexts
            enh_cont_df['Normalized CRE effect'] = enh_cont_df[
                '(MUT - CONTROL) / WT']  # different norm for tiles from enh vs sil
            sil_cont_df['Normalized CRE effect'] = sil_cont_df['(MUT - CONTROL) / CONTROL']

            enh_cres = (enh_cont_df[(enh_cont_df['Normalized CRE effect'] > 0.3)]).copy()
            enh_cres['tile class'] = 'Enhancer'
            selected_cres.append(enh_cres)
            sil_cres = (sil_cont_df[(sil_cont_df['Normalized CRE effect'] < -0.3)]).copy()
            sil_cres['tile class'] = 'Silencer'
            selected_cres.append(sil_cres)
        selected_cres = pd.concat(selected_cres)
        selected_cres.to_csv(f'{csv_dir}/sufficient_CREs.csv')

if __name__ == '__main__':
    main()



