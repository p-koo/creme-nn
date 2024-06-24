import glob
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import sys, os
from tqdm import tqdm
import json
import copy

from creme import creme
from creme import custom_model
from creme import utils


def process_greedy_search_res(res_path, cre_tiles, log, optimization_name):
    raw_res = utils.read_pickle(res_path)
    res = {}
    if log:
        for i in raw_res.keys():
            res[i] = {}
            for k, v in raw_res[i].items():
                if 'pred' in k:
                    res[i][k] = np.log(v)
                else:
                    res[i][k] = v
    else:
        res = raw_res

    wt = res[0]['initial_pred']
    trace = [res[i]['initial_pred'] / wt for i in res.keys()]
    if optimization_name == 'min':
        trace.append(np.min(res[24]['preds'].mean(axis=-1)) / wt)
    else:
        trace.append(np.max(res[24]['preds'].mean(axis=-1)) / wt)

    ####hypothetical additive model
    greedy_search_order = [np.argwhere(np.array(cre_tiles) == res[i]['selected_tile'])[0][0] for i in
                           res.keys()]

    mutant_predictions_first_iter = res[0]['preds'].mean(axis=-1)

    effect_sizes_first_iter = mutant_predictions_first_iter - wt
    sorted_effect_first_iter = effect_sizes_first_iter[greedy_search_order]
    sorted_effect = (sorted_effect_first_iter / wt)
    sum_of_effects = np.cumsum(sorted_effect_first_iter)
    hypothetical_trace = wt + sum_of_effects
    hypothetical_trace = (np.concatenate([[wt], hypothetical_trace]) / wt)#[:-1] # last of greedy is useless
    return [trace, hypothetical_trace, sorted_effect]

def main():
    model_name = sys.argv[1]
    optimization_name = sys.argv[2]

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
    csv_dir = f'../results/summary_csvs/{model_name}/'
    cell_lines = []
    for track_index in [4824, 5110, 5111]:
        print(f'USING model {model_name}')
        if model_name.lower() == 'enformer':
            bin_index = [447, 448]
            model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
            target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
            cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])
            cell_lines.append(cell_line)
            num_rounds = 25
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Optimizing for cell line:')
            print(cell_line)
        else:
            print('Unkown model')
            sys.exit(1)

        result_dir_cell = utils.make_dir(f'{result_dir_model}/{cell_line}')

        context_df = pd.read_csv(f'{csv_dir}/{cell_line}_selected_contexts.csv')

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
                result_summary = creme.higher_order_interaction_test(model, x, copy.copy(cre_tiles), optimization, num_shuffle,
                                                                     num_rounds)

                utils.save_pickle(result_path, result_summary)

    locations_all = []
    second_its_all = []
    traces_all = []

    for cell_line in cell_lines:
        # location_map = [0 for _ in range(len(cre_tiles))]
        context_df = pd.read_csv(f'../results/summary_csvs/{model_name}/{cell_line}_selected_contexts.csv')

        print(context_df.shape)
        for _, row in context_df.iterrows():
            res_path = f"../results/higher_order_test_{optimization_name}/{model_name}/{cell_line}/{row['path'].split('/')[-1]}"
            res = utils.read_pickle(res_path)
            greedy_search_order = [np.argwhere(np.array(cre_tiles) == res[i]['selected_tile'])[0][0] for i in
                                   res.keys()]
            first_two_points = res[0]['preds'][greedy_search_order[:2]].mean(axis=1)
            two_cres_shuffled = res[2]['initial_pred']
            second_it = [res[0]['initial_pred'], first_two_points[0], first_two_points[1], two_cres_shuffled]


            trace, hypothetical_trace, sorted_effect = process_greedy_search_res(res_path, cre_tiles, False,
                                                                                 optimization_name)
            log_trace, log_hypothetical_trace, log_sorted_effect = process_greedy_search_res(res_path, cre_tiles, True,
                                                                                             optimization_name)


            location_map = [0 for _ in range(len(cre_tiles))]
            for iteration in range(5):
                location_map[np.argwhere(res[iteration]['selected_tile'] == np.array(cre_tiles))[0][0]] = 1
            location_map = pd.DataFrame(location_map, columns=['tile_selected'])
            location_map['context'] = row['context']
            location_map['seq_id'] = row['seq_id']
            location_map['cell_line'] = cell_line
            locations_all.append(location_map)

            one_seq_res = pd.DataFrame([trace, hypothetical_trace, sorted_effect,
                                        log_trace, log_hypothetical_trace, log_sorted_effect]).T
            one_seq_res.columns = ['trace', 'hypothetical_trace', 'sorted_effects',
                                   'log_trace', 'log_hypothetical_trace', 'log_sorted_effects']
            one_seq_res['cell_line'] = cell_line
            one_seq_res['context'] = row['context']
            one_seq_res['seq_id'] = row['seq_id']
            traces_all.append(one_seq_res)

            second_it_points = pd.DataFrame(second_it)
            second_it_points['context'] = row['context']
            second_it_points['seq_id'] = row['seq_id']
            second_it_points['cell_line'] = cell_line
            second_its_all.append(second_it_points.set_index(pd.Index(['wt', 'greedy_it1', 'greedy_it2',
                                                                       'two_cres_shuffled'])))


    greedy_csv_dir = utils.make_dir(f'../results/summary_csvs/{model_name}/greedy_search/')
    pd.concat(second_its_all).to_csv(f'{greedy_csv_dir}/{optimization_name}_second_iteration.csv')
    pd.concat(traces_all).to_csv(f'{greedy_csv_dir}/{optimization_name}_traces.csv')
    pd.concat(locations_all).to_csv(f'{greedy_csv_dir}/{optimization_name}_locations.csv')

if __name__ == '__main__':
    main()
