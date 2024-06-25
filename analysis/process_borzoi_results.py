from creme import utils
import sys
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm



def main():


    #######CONTEXT DEPENDENCE TEST
    print("Processing context dependence test results")
    model_name = 'borzoi'
    cell_lines = ['K562 ENCODE, biol_', 'GM12878 ENCODE, biol_', 'PC-3']
    target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
    cell_line_info, cage_tracks = utils.get_borzoi_targets(target_df, cell_lines)

    summary_csv_dir = f'../results/summary_csvs/{model_name}'
    threshold_enh, threshold_neu, threshold_sil = 0.9, 0.05, -0.2
    per_model_delta = {}
    res = []
    for i, c in enumerate(cell_lines):
        per_model_delta[c] = []
        deltas = []
        wt_means = []
        seq_ids = []
        selected_tss = pd.read_csv(glob.glob(f'../results/summary_csvs/borzoi/*_{c}*selected_genes.csv')[0])
        for _, row in selected_tss.iterrows():
            path = f'../results/context_dependence_test_5/{model_name}/{utils.get_summary(row)}.pickle'
            context_res = utils.read_pickle(path)
            tss_activity = utils.read_pickle(
                f'../results/gencode_tss_predictions//{model_name}/{utils.get_summary(row)}.pickle')

            wt = ((tss_activity.mean(axis=0).sum(axis=-2) / 2))[:, cell_line_info[c]['output']].sum(axis=-1)
            mut = ((context_res['mut'].mean(axis=0).sum(axis=-2) / 2))[:, cell_line_info[c]['output']].sum(axis=-1)
            delta_per_model = (wt - mut) / wt
            per_model_delta[c].append(delta_per_model)
            deltas.append(delta_per_model.mean())
            wt_means.append(wt.mean())
            seq_ids.append(path)

        df = pd.DataFrame([deltas, seq_ids]).T
        df.columns = ['delta_mean', 'path']
        df['wt'] = wt_means
        df['context'] = [v for v in pd.cut(df['delta_mean'],
                                           [df['delta_mean'].min() - 1, threshold_sil, -threshold_neu, threshold_neu,
                                            threshold_enh, df['delta_mean'].max() + 1],
                                           labels=['silencing', 'other1', 'neutral', 'other', 'enhancing']).values]
        df['cell_line'] = c.split()[0]

        print(df.shape)
        res.append(df)

    res = pd.concat(res)

    res.to_csv(f'{summary_csv_dir}/context_dependence_test.csv')

    ######## SUFFICIENCY TEST

    context_dfs_per_cell = {}
    for cell_line in cell_lines:
        df = pd.read_csv(f'../results/summary_csvs/borzoi/{cell_line.split()[0]}_selected_contexts.csv')
        df['seq_id'] = [r.split('.')[-2].split('/')[-1] for r in df['path']]

        context_dfs_per_cell[cell_line.split()[0]] = df
    tile_df = pd.read_csv('../results/summary_csvs/borzoi/sufficiency_test_tile_coordinates.csv').T.iloc[1:-1, :]
    cell_lines_dict = {'K562': 'K562 ENCODE, biol_', 'GM12878': 'GM12878 ENCODE, biol_', 'PC-3': 'PC-3'}
    print('Processing Sufficiency test results')

    result_summary = []
    for cell_line, df in context_dfs_per_cell.items():
        track_index = cell_line_info[cell_lines_dict[cell_line]]['output']
        for i, row in df.iterrows():
            res = utils.read_pickle(f'../results/sufficiency_test/borzoi/{row["seq_id"]}.pickle')
            res['wt'] = (res['wt'].sum(axis=0) / 2)[track_index].sum()
            res['control'] = res['control'][:, :, track_index].sum(axis=-1).sum(axis=-1) / 2
            res['mut'] = res['mut'][:, :, track_index].sum(axis=-1).sum(axis=-1) / 2

            one_seq = pd.DataFrame((res['mut'] - res['control']) / res['wt'])
            one_seq.columns = ['(MUT - CONTROL) / WT']
            one_seq['(MUT - CONTROL) / CONTROL'] = (res['mut'] - res['control']) / res['control']
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
    result_summary.to_csv(f'{summary_csv_dir}/sufficiency_test.csv')


    ######## DISTANCE TEST
    print('Processing distance test results')
    borzoi_model_seq_length = 524288
    cre_df = pd.read_csv(f'../results/summary_csvs/borzoi/sufficient_CREs.csv')
    tss_tile, cre_tiles = utils.set_tile_range(borzoi_model_seq_length, 5000)

    cre_tiles_starts = np.array(cre_tiles).T[0]
    cre_tiles_starts_abs = (cre_tiles_starts - tss_tile[0]) // 1000
    cre_df.insert(1, "distance to TSS (Kb)",
                  [(int(i) - tss_tile[0]) // 1000 for i in cre_df['tile_start'].values])
    test_results = f'../results/distance_test_True/borzoi_5/'
    target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
    cell_lines_for_search = ['K562 ENCODE, biol_']

    cell_line_info, track_index = utils.get_borzoi_targets(target_df, cell_lines_for_search)
    normalized_tests = []
    for j, (_, row) in tqdm(enumerate(cre_df.iterrows())):
        tile_start, tile_end = [row['tile_start'], row['tile_end']]
        result_path = f'{test_results}/{row["seq_id"]}_{tile_start}_{tile_end}.pickle'
        res = utils.read_pickle(result_path)
        test = (res['mut'].mean(axis=1).sum(axis=1)/2).sum(axis=-1)
        control = row['control']
        if row['context'] == 'enhancing':
            CRE_norm_effects = (test - control) / row['wt']
        else:
            CRE_norm_effects = (test - control) / control
        # norm_effects = test / control
        norm_effects = test / np.max(test)
        # normalized_tests.append(norm_effects)
        df = pd.DataFrame([norm_effects, CRE_norm_effects, cre_tiles_starts_abs]).T
        df.columns = ['Fold change over control', 'CRE sufficiency effect', 'Binned distance (Kb)']


        df['tile class'] = row['tile class']
        df['cell line'] = 'K562'
        df['seq_id'] = f"{row['seq_id']}_{row['tile_start']}_{row['tile_end']}"
        normalized_tests.append(df)

    result_normalized_effects = pd.concat(normalized_tests)
    result_normalized_effects.to_csv(f'{summary_csv_dir}/distance_test.csv')


if __name__ == "__main__":
    main()