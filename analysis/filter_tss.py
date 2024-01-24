import pandas as pd
import numpy as np
import sys
sys.path.append('../creme/')
import custom_model
import utils
import glob
from tqdm import tqdm



def main():
    model_name = sys.argv[1]
    tss_df = pd.read_csv('../results/tss_positions.csv')
    result_dir = f'../results/gencode_tss_predictions/{model_name}'
    target_df = pd.read_csv(f'../data/{model_name}_targets_human.txt', sep='\t')


    if model_name == 'enformer':
        cell_lines = [4824, 5110, 5111]
        bin_index = [447, 448]
        print(f'Using bins {bin_index}')


        column_names = [t.split(':')[-1].split(' ENCODE')[0].strip() for t in target_df.iloc[cell_lines]['description'].values]
        print(column_names)
    elif model_name == 'borzoi':
        cell_lines = ['K562 ENCODE, biol_', 'GM12878 ENCODE, biol_', 'PC-3']
        column_names = cell_lines
        cell_line_info = utils.get_borzoi_targets(target_df, cell_lines)


    N = tss_df.shape[0]
    all_tss = np.empty((N, len(cell_lines)))
    for i, row in tqdm(tss_df.iterrows(), total=N):
        if model_name == 'enformer':
            pred = np.load(f'{result_dir}/{utils.get_summary(row)}.npy')[:, cell_lines]
            all_tss[i] = pred[bin_index].mean(axis=0)
        elif model_name == 'borzoi':


            pred = utils.read_pickle(f'{result_dir}/{utils.get_summary(row)}.pickle')
            pred = (pred[0, :, :, :].mean(axis=0).sum(axis=0) / 2)
            for j, (cell_line, v) in enumerate(cell_line_info.items()):

                indeces = v['output']
                all_tss[i, j] = pred[indeces].mean() # sum across strands of cell line tracks



    np.save(f'../results/{model_name}_summary_cage.npy', all_tss)
    output_dir = utils.make_dir(f"{utils.make_dir(f'../results/summary_csvs/')}/{model_name}/")

    for i in range(len(cell_lines)):
        if model_name == 'enformer':
            save_path = f'{output_dir}/{cell_lines[i]}_{column_names[i]}_selected_genes.csv'
        elif model_name == 'borzoi':
            save_path = f'{output_dir}/{cell_line_info[cell_lines[i]]["target"]}_{cell_lines[i]}_selected_genes.csv'
        print(save_path)
        cell_line_df = tss_df.copy()
        cell_line_df[column_names[i]] = all_tss[:, i]

        max_tss_set = cell_line_df.sort_values(column_names[i], ascending=False).drop_duplicates(['gene_name'])

        max_tss_set = max_tss_set.sort_values(column_names[i]).iloc[-10000:]

        max_tss_set.to_csv(save_path)


if __name__ == "__main__":
    main()