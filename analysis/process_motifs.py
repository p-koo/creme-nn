import pandas as pd
import numpy as np
import glob
import sys, os
sys.path.append('../creme')
import utils, custom_model
from tqdm import tqdm

def main():
    summary_csv_dir = utils.make_dir('../results/summary_csvs/enformer/motif_analysis')
    xstreme_res_dir = '../results/XSTREME/FIMO/'
    saliency_dir = "../results/saliency"
    result_dir = '../results/motifs_500,50_batch_1,10_shuffle_10_thresh_0.9,0.7/'

    #
    ######## XSTREME VS SALIENCY VS CREME
    bps = np.arange(0, 5001, 500)
    dfs = []
    for a, (index, cell_line) in enumerate({4824: 'PC-3', 5110: 'GM12878', 5111: 'K562'}.items()):
        for s, seq_tile_id in enumerate(os.listdir(f'{saliency_dir}/{index}/')):
            # CREME results
            prune_res = utils.read_pickle(f'{result_dir}/{cell_line}/{seq_tile_id}')
            creme_frac = np.array([1] + prune_res[500]['scores'][:-1] + prune_res[50]['scores'])
            creme_bps = bps[:len(creme_frac)]
            control = prune_res['mut']

            # XSTREME
            xstreme_res = utils.read_pickle(
                glob.glob(f'{xstreme_res_dir}/{cell_line}_enhancers_*/{seq_tile_id}')[0])

            # Saliency
            saliency_res = utils.read_pickle(f'{saliency_dir}/{index}/{seq_tile_id}')
            saliency_frac = np.array(saliency_res['saliency']) / control
            random_frac = np.array(saliency_res['random']) / control
            df = pd.DataFrame([creme_frac, creme_bps, saliency_frac, random_frac]).T
            df.columns = ['CREME score', 'CREME bps', 'Saliency score', 'Random score']
            df['XSTREME score'] = xstreme_res['motifs'] / control
            df['seq_id'] = seq_tile_id
            df['cell_line'] = cell_line
            dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(f'{summary_csv_dir}/CREME_vs_saliency_vs_XSTREME.csv')

    #### EXAMPLES
    model = custom_model.Enformer()
    seq_parser = utils.SequenceParser('../data/GRCh38.primary_assembly.genome.fa')

    cell_line = 'K562'
    track_index = 5111
    dfs = []
    for filename in os.listdir(f'../results/saliency/{track_index}/'):
        if filename.split('_')[0] in ['GATA2', 'MYEOV', 'GAD1',
                                      'CPNE3', 'FMNL1', 'MYADM']:
            cre_saliency_scores, creme_mask = utils.get_saliency_and_creme_mask_overlap(filename, model, seq_parser,
                                                                                        cell_line,
                                                                                        track_index)
            xstreme_res = \
            utils.read_pickle(glob.glob(f'{xstreme_res_dir}/{cell_line}_enhancers_*/{filename}')[0])['motif_mask']

            df = pd.DataFrame([cre_saliency_scores.numpy(), creme_mask, xstreme_res]).T
            df.columns = ['Saliency', 'CREME', 'XSTREME']
            df['gene'] = filename.split('_')[0]
            dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(f"{summary_csv_dir}/example_seqs.csv")

if __name__ == '__main__':
    main()