import glob
import pickle
import pandas as pd
import numpy as np
import sys, os
from tqdm import tqdm
import json
from creme import creme
from creme import custom_model
from creme import utils
from creme import shuffle

def main():
    model_name = sys.argv[1]
    # track_index = int(sys.argv[2])
    optimization_name = sys.argv[2]

    perturb_window = 5000
    num_shuffle = 10
    data_dir = '../data/'
    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    higher_order_test_result_dir = f'../results/higher_order_test_{optimization_name}/{model_name}' # existing results
    result_dir_model = utils.make_dir(f'{higher_order_test_result_dir}/sufficiency/')
    csv_dir = f'../results/summary_csvs/{model_name}/'

    for track_index in [4824, 5110, 5111]:
        print(f'USING model {model_name}')
        if model_name.lower() == 'enformer':
            bin_index = [447, 448]
            model = custom_model.Enformer(track_index=track_index, bin_index=bin_index)
            target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
            cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Testing sufficiency for cell line:')
            print(cell_line)
        else:
            print('Unknown model')
            sys.exit(1)


        result_dir_cell = utils.make_dir(f'{result_dir_model}/{cell_line}')

        context_df = pd.read_csv(f'{csv_dir}/{cell_line}_selected_contexts.csv')

        context_df = context_df.sample(frac=1)


        # get coordinates of central tss
        tss_tile, cre_tiles = utils.set_tile_range(model.seq_length, perturb_window)
        # set up sequence parser from fasta
        seq_parser = utils.SequenceParser(fasta_path)

        for i, row in tqdm(context_df.iterrows(), total=len(context_df)):
            seq_id = row['path'].split('/')[-1].split('.')[0]
            result_path = f'{result_dir_cell}/{seq_id}.pickle'
            print(result_path)

            if not os.path.isfile(result_path):
                result_summary = {}
                chrom, start, strand = seq_id.split('_')[1:]
                # get seq from reference genome and convert to one-hot
                x = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length, onehot=True)
                only_tss_seqs = shuffle.dinuc_shuffle(x, num_shuffle)
                only_tss_seqs[:, tss_tile[0]:tss_tile[1], :] = x[tss_tile[0]:tss_tile[1], :].copy()
                result_summary['only_tss_preds'] = model.predict(only_tss_seqs).mean()
                current_seq = only_tss_seqs.copy()
                result_summary['predictions'] = []
                result_summary['tile_added'] = []

                old_res = utils.read_pickle(f'{higher_order_test_result_dir}/{cell_line}/{seq_id}.pickle')
                for i in tqdm(old_res.keys()):
                    tile_start, tile_end = old_res[i]['selected_tile']
                    current_seq[:, tile_start: tile_end, :] = x[tile_start: tile_end, :].copy()
                    result_summary['predictions'].append(model.predict(current_seq).mean())
                    result_summary['tile_added'].append(old_res[i]['selected_tile'])
                utils.save_pickle(result_path, result_summary)


    ######SUMMARIZE
    if optimization_name == 'min':
        element_type = 'Enhancers'
        context_type = 'enhancing'
    elif optimization_name == 'max':
        element_type = 'Silencers'
        context_type = 'silencing'

    traces = []

    for cell_line in ['K562', 'GM12878', 'PC-3']:
        result_dir = f'../results/higher_order_test_{optimization_name}/enformer/sufficiency/{cell_line}/'
        context_df = pd.read_csv(f'../results/summary_csvs/enformer/{cell_line}_selected_contexts.csv')
        context_df = context_df[context_df['context'] == context_type]
        for i, row in context_df.iterrows():

            seq_id = row['path'].split('/')[-1].split('.')[0]
            result_path = f'{result_dir}/{seq_id}.pickle'
            res = utils.read_pickle(result_path)
            res_greedy = utils.read_pickle(result_path.replace('sufficiency', ''))
            wt = res_greedy[0]['initial_pred']
            res['predictions'].insert(0, res['only_tss_preds'])
            preds = np.array(res['predictions'])
            # trace = preds / wt
            if element_type == 'Enhancers':
                trace = preds / wt

            else:
                trace = preds / res['only_tss_preds']
            df = pd.DataFrame(trace, columns=['Normalized TSS activity'])
            df['context'] = row['context']
            df['cell_line'] = cell_line
            df['seq_id'] = row['seq_id']
            traces.append(df)
    traces = pd.concat(traces)
    traces.to_csv(f"{csv_dir}/greedy_search/sufficiency_of_greedy_tiles_{optimization_name}.csv")


if __name__ == '__main__':
    main()

