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

    print(f'USING model {model_name}')
    if model_name.lower() == 'enformer':
        model = custom_model.Enformer(track_index=[4824, 5110, 5111])
    elif model_name.lower() == 'borzoi':
        target_df = pd.read_csv('../data/borzoi_targets_human.txt', sep='\t')
        cage_rna_tracks = [i for i, t in enumerate(target_df['description']) if 'CAGE' in t or 'RNA' in t]
        model = custom_model.Borzoi('../data/borzoi/*/*', cage_rna_tracks, True)

    else:
        print('Unkown model')
        sys.exit(1)

    data_dir = '../data/'
    result_dir = f'../results/'
    test_results_dir = utils.make_dir(f'{result_dir}/context_dependence_test')

    fasta_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    seq_parser = utils.SequenceParser(fasta_path)

    tss_df = pd.concat(pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob(f'{result_dir}/gencode_tss_predictions/{model_name}/*selected_tss.csv'))
    tss_df = tss_df.iloc[:, :-3].drop_duplicates()
    model_results_dir = utils.make_dir(f'{test_results_dir}/{model_name}/')

    tss_df = tss_df.sample(frac=1)
    seq_halflen = model.seq_length // 2
    half_window_size = 5000 // 2
    N_shuffles = 10
    for i, row in tqdm(tss_df.iterrows(), total=tss_df.shape[0]):
        result_path = f"{model_results_dir}/{utils.get_summary(row)}.pickle"
        if not os.path.isfile(result_path):
            x = seq_parser.extract_seq_centered(row['Chromosome'], row['Start'], row['Strand'], model.seq_length)

            pred_wt, pred_mut, pred_std = creme.context_dependence_test(model, x,
                                                                        [seq_halflen - half_window_size, seq_halflen + half_window_size],
                                                                        N_shuffles)

            with open(result_path, 'wb') as handle:
                pickle.dump({'wt': pred_wt, 'mut': pred_mut, 'std': pred_std},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__=='__main__':
    main()



