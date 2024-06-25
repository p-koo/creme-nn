import pandas as pd
import numpy as np
import sys
import glob
from tqdm import tqdm
import os
from creme import creme
from creme import custom_model
from creme import utils

def main():
    track_index = int(sys.argv[1])

    model_name = 'enformer'
    data_dir = '../data'
    seq_parser = utils.SequenceParser(f'{data_dir}/GRCh38.primary_assembly.genome.fa')
    target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
    print(track_index, type(track_index))
    cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])
    print(f'COMPUTING SALIENCY FOR CELL LINE {cell_line}')
    model = custom_model.Enformer()

    result_dir = utils.make_dir(f'../results/saliency/')
    result_dir = utils.make_dir(f"{result_dir}/{track_index}/")
    csv_dir = f'../results/summary_csvs/{model_name}/'
    cre_df_path = f'{csv_dir}/sufficient_CREs.csv'
    all_cre_set = pd.read_csv(cre_df_path)
    all_cre_set = all_cre_set[(all_cre_set['cell_line'] == cell_line) & (all_cre_set['tile class'] == 'Enhancer')]
    bps = np.arange(0, 5001, 500)
    target_bins = [447, 448]

    for cell_line, cre_set in all_cre_set.groupby('cell_line'):

        for r, row in tqdm(cre_set.iterrows(), total=cre_set.shape[0]):
            tile_start, tile_end = row['tile_start'], row['tile_end']
            result_path = f"{result_dir}/{row['seq_id']}_{row['tile_start']}_{row['tile_end']}.pickle"
            prune_res_path = f"../results/motifs_500,50_batch_1,10_shuffle_10_thresh_0.9,0.7/{cell_line}/{row['seq_id']}_{row['tile_start']}_{row['tile_end']}.pickle"
            prune_res = utils.read_pickle(prune_res_path)
            if (not os.path.isfile(result_path)) and (500 in prune_res.keys()):
                print(result_path)
                control_sequences = prune_res['control_sequences']

                chrom, start, strand = row['seq_id'].split('_')[1:]
                wt_seq = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length)
                wt_seq_padded = np.pad(wt_seq[np.newaxis].copy(),
                                       ((0, 0), (model.seq_length // 2, model.seq_length // 2), (0, 0)), 'constant')
                predictions = model.predict(wt_seq[np.newaxis])[0]
                target_mask = np.zeros_like(predictions)
                for idx in target_bins:
                    target_mask[idx, track_index] = 1

                cre_saliency_scores = model.contribution_input_grad(wt_seq_padded, target_mask)[
                           model.seq_length // 2:-model.seq_length // 2][tile_start: tile_end]


                abs_saliency_positions = [l + row['tile_start'] for l in np.argsort(np.abs(cre_saliency_scores))]
                saliency_positions = [l + row['tile_start'] for l in np.argsort(cre_saliency_scores)]
                random_pos = [l + row['tile_start'] for l in
                              np.random.choice(list(range(tile_end - tile_start)), 5000, replace=False)]
                result_summary = {}
                for label, mask in {'saliency': saliency_positions, 'abs_saliency': abs_saliency_positions,
                                    'random': random_pos}.items():
                    preds = []
                    for bp in bps:
                        current_mask = mask.copy()[:bp]
                        prune_seqs = control_sequences.copy()
                        prune_seqs[:, tile_start: tile_end, :] = wt_seq[tile_start:tile_end].copy()
                        prune_seqs[:, current_mask, :] = control_sequences[:, current_mask, :].copy()

                        preds.append(model.predict(prune_seqs)[:, target_bins, track_index].mean())
                    result_summary[label] = preds
                utils.save_pickle(result_path, result_summary)

if __name__ == "__main__":
    main()