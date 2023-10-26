import pandas as pd
import numpy as np
import sys
sys.path.append('../creme/')
import custom_model
import utils
import glob
from tqdm import tqdm



def main():
    tss_df = pd.read_csv('../results/tss_positions.csv')
    cell_lines = [4824, 5110, 5111]
    bin_index = [446, 447, 448, 449]
    print(f'Using bins {bin_index}')
    model_name = 'enformer'
    result_dir = f'../results/gencode_tss_predictions/{model_name}'
    targets = pd.read_csv(f'../data/{model_name}_targets_human.txt', sep='\t')
    column_names = [t.split(':')[-1].split(' ENCODE')[0].strip() for t in targets.iloc[cell_lines]['description'].values]
    print(column_names)

    N = tss_df.shape[0]
    all_tss = np.empty((N, len(cell_lines)))
    for i, row in tqdm(tss_df.iterrows(), total=N):
        pred = np.load(f'{result_dir}/{i}.npy')[:, cell_lines]
        all_tss[i] = pred[bin_index].mean(axis=0)

    np.save(f'../results/{model_name}_summary_cage.npy', all_tss)

    for i in range(len(cell_lines)):
        cell_line_df = tss_df.copy()
        cell_line_df[column_names[i]] = all_tss[:, i]

        max_tss_set = cell_line_df.sort_values(column_names[i], ascending=False).drop_duplicates(['gene_name'])

        max_tss_set = max_tss_set.sort_values(column_names[i]).iloc[-5000:]

        max_tss_set.to_csv(f'{result_dir}/{cell_lines[i]}_{column_names[i]}_selected_tss.csv')


if __name__ == "__main__":
    main()