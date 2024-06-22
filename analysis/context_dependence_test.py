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
    N_shuffles = int(sys.argv[2])
    threshold_enh, threshold_neu, threshold_sil = 0.95, 0.05, -0.3
    half_window_size = 5000 // 2
    max_sample_size = 200 # max number of each context type to choose

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        bin_index = [447, 448] # save this but keep all results for visualization

        track_index = [4824, 5110, 5111]
        model = custom_model.Enformer(track_index=track_index)
        target_df = pd.read_csv(f'../data/enformer_targets_human.txt', sep='\t')
        cell_line_info = {i: [t, utils.clean_cell_name(target_df.iloc[t]['description'])] for i, t in
                      enumerate(track_index)}
    elif model_name.lower() == 'borzoi':
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
        model = custom_model.Borzoi('../data/borzoi/*/*', track_index=track_index, aggregate=False)
        model.bin_index = list(np.arange(model.target_lengths // 2 - 4, model.target_lengths // 2 + 4, 1))

    else:
        print('Unkown model')
        sys.exit(1)

    # genome path
    fasta_path = f'../data/GRCh38.primary_assembly.genome.fa'
    result_dir = f'../results/' # base dir for results
    csv_dir = f'{result_dir}/summary_csvs/{model_name}' # dir with all summary csvs
    model_results_dir = utils.make_dir(f"{utils.make_dir(f'{result_dir}/context_dependence_test_{N_shuffles}')}/{model_name}") # output of this test
    print(model_results_dir)
    selected_gene_csvs = glob.glob(f'{csv_dir}/*selected_genes.csv')


    seq_parser = utils.SequenceParser(fasta_path)
    tss_df = pd.concat(pd.read_csv(f, index_col='Unnamed: 0') for f in selected_gene_csvs) # load selected gene lists
    tss_df = tss_df.iloc[:, :-3].drop_duplicates()
    tss_df = tss_df.sample(frac=1)
    seq_halflen = model.seq_length // 2

    #
    for i, row in tqdm(tss_df.iterrows(), total=tss_df.shape[0]):
        result_path = f"{model_results_dir}/{utils.get_summary(row)}.pickle"
        if not os.path.isfile(result_path):
            x = seq_parser.extract_seq_centered(row['Chromosome'], row['Start'], row['Strand'], model.seq_length)
            if model_name == 'enformer':
                pred_wt, pred_mut, pred_std = creme.context_dependence_test(model, x,
                                                                            [seq_halflen - half_window_size, seq_halflen + half_window_size],
                                                                            N_shuffles)

                with open(result_path, 'wb') as handle:
                    pickle.dump({'wt': pred_wt, 'mut': pred_mut, 'std': pred_std},
                                handle, protocol=pickle.HIGHEST_PROTOCOL)

            elif model_name == 'borzoi':
                _, pred_mut = creme.context_dependence_test(model, x,
                                                            [seq_halflen - half_window_size, seq_halflen + half_window_size],
                                                            N_shuffles, mean=False, drop_wt=True)

                with open(result_path, 'wb') as handle:
                    pickle.dump({'mut': pred_mut},
                                handle, protocol=pickle.HIGHEST_PROTOCOL)

    ####### SUMMARIZE RESULTS
    if model_name == 'enformer':

        summary_combined = []
        for i, (cell_index, cell_name) in cell_line_info.items():
            print(f'Processing results from {cell_name}')
            if model_name == 'enformer':
                selected_tss = pd.read_csv(f'{csv_dir}/{cell_index}_{cell_name}_selected_genes.csv')
            elif model_name == 'borzoi':
                cell_name = i.split()[0]
                print(f'{csv_dir.replace("borzoi", "enformer")}/*_{cell_name}_selected_genes.csv')
                selected_tss = pd.read_csv(glob.glob(f'{csv_dir}/*_{cell_name}_selected_genes.csv')[0])

            summary_per_cell = {k: [] for k in ['delta_mean', 'path', 'wt', 'std', 'mean_mut', 'seq_id']}
            for _, row in selected_tss.iterrows():
                path = f'{model_results_dir}/{utils.get_summary(row)}.pickle'
                summary_per_cell['path'].append(path)
                summary_per_cell['seq_id'].append(utils.get_summary(row))
                with open(path, 'rb') as handle:
                    context_res = pickle.load(handle)
                delta = creme.context_effect_on_tss(context_res['wt'][bin_index, i].mean(),
                                                    context_res['mut'][bin_index, i].mean())
                summary_per_cell['delta_mean'].append(delta)
                summary_per_cell['wt'].append(context_res['wt'][bin_index, i].mean())
                summary_per_cell['std'].append(context_res['std'][bin_index, i].mean())
                summary_per_cell['mean_mut'].append(context_res['mut'][bin_index, i].mean())

            summary_per_cell = pd.DataFrame.from_dict(summary_per_cell)
            summary_per_cell['context'] = [v for v in pd.cut(summary_per_cell['delta_mean'],
                                                             [summary_per_cell['delta_mean'].min() - 1, threshold_sil,
                                                              -threshold_neu, threshold_neu, threshold_enh,
                                                              summary_per_cell['delta_mean'].max() + 1],
                                                             labels=['silencing', 'other1', 'neutral', 'other',
                                                                     'enhancing']).values]
            summary_per_cell['cell_line'] = cell_name

            print(summary_per_cell.shape)
            summary_combined.append(summary_per_cell)

        summary_combined = pd.concat(summary_combined).reset_index(drop=True)
        summary_combined.to_csv(f'{csv_dir}/context_dependence_test.csv') # summary of context effect for every selected gene


        ####### SELECT CONTEXTS - ENHANCING, NEUTRAL, SILENCING


        for k, df in summary_combined.groupby('cell_line'):  # per cell line
            context_df = df[(df['context'] != 'other') & (df['context'] != 'other1')]  # remove unclassified contexts
            context_df_subsample = []

            for context_type, one_context_df in context_df.groupby('context'):
                # subset if more than threshold number of rows
                if one_context_df.shape[0] > max_sample_size:
                    context_df_subsample.append(one_context_df.sample(max_sample_size, random_state=42))
                else:
                    context_df_subsample.append(one_context_df)
            context_df = pd.concat(context_df_subsample)
            context_df.to_csv(f'{csv_dir}/{k}_selected_contexts.csv')

        k = 'K562'
        cell_line_names = ['PC-3', 'GM12878', 'K562']

        df = pd.read_csv(f'../results/summary_csvs/enformer/{k}_selected_contexts.csv').sort_values('context')
        cell_pred_index = cell_line_names.index(k)
        example_set = []
        for name, selected_point in zip(['enhancing context', 'neutral context', 'silencing context'],
                                        [df[df['context'] == 'enhancing'].sort_values('delta_mean')['path'].values[-1],
                                         df[df['context'] == 'neutral']['path'].values[0],
                                         df[df['context'] == 'silencing'].sort_values('delta_mean')['path'].values[0]
                                         ]):
            with open(selected_point, 'rb') as handle:
                context_res = pickle.load(handle)
            wt = context_res['wt'][:, cell_pred_index]
            mut = context_res['mut'][:, cell_pred_index]
            one_example = pd.DataFrame([wt, mut]).T
            one_example.columns = ['WT', 'MUT']
            one_example['name'] = name
            example_set.append(one_example)
        example_set = pd.concat(example_set)
        example_set.to_csv(f'../results/summary_csvs/enformer/example_sequences.csv')

        ### save profiles
        cell_lines = [4824, 5110, 5111]
        cell_line_names = ['PC-3', 'GM12878', 'K562']
        bin_index = [447, 448]
        model_name = 'enformer'
        summary_dfs = []
        for i, c in enumerate(cell_lines):
            selected_tss_path = \
            glob.glob(f'../results/summary_csvs/{model_name}/{c}_{cell_line_names[i]}*_selected_genes.csv')[0]
            cell_name = selected_tss_path.split('_')[-3]
            wts = []
            selected_tss = pd.read_csv(selected_tss_path)
            for _, row in tqdm(selected_tss.iterrows()):
                r = f'../results/context_dependence_test_100///{model_name}/{utils.get_summary(row)}.pickle'

                with open(r, 'rb') as handle:
                    context_res = pickle.load(handle)
                wts.append(context_res['wt'][:, i])

            window = 10
            center_coverage = np.array(wts).mean(axis=0)[448 - window: 448 + window]
            std = np.array(wts).std(axis=0)[448 - window: 448 + window]
            df = pd.DataFrame([center_coverage, std]).T
            df.columns = ['mean', 'std']
            df['cell_line'] = cell_name
            summary_dfs.append(df)
        pd.concat(summary_dfs).to_csv('../results/summary_csvs/enformer/coverage_profiles.csv')

if __name__=='__main__':
    main()



