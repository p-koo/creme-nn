import sys
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import tempfile
import matplotlib.pyplot as plt
import kipoiseq
from tqdm import tqdm
from pymemesuite.fimo import FIMO
from pymemesuite.common import MotifFile
import Bio.SeqIO
from pymemesuite.common import Sequence
import copy

sys.path.append('../creme')
import creme
import custom_model
import utils


def main():

    meme_path = sys.argv[1]
    track_index = int(sys.argv[2])

    data_dir = '../data/'
    genome_path = f'{data_dir}/GRCh38.primary_assembly.genome.fa'
    seq_parser = utils.SequenceParser(genome_path)
    model_name = 'enformer'
    meme_filename = meme_path.split("/")[-1].split('.')[0]
    model = custom_model.Enformer(track_index=track_index, bin_index=[447, 448])
    target_df = pd.read_csv(f'{data_dir}/enformer_targets_human.txt', sep='\t')
    print(track_index)
    cell_line = utils.clean_cell_name(target_df.iloc[track_index]['description'])
    print(f'USING CELL LINE {cell_line} with meme file {meme_filename}')

    result_dir = utils.make_dir(f"{utils.make_dir(f'../results/XSTREME/')}/FIMO/")
    result_dir = utils.make_dir(f"{result_dir}/{meme_filename}_{track_index}/")
    csv_dir = f'../results/summary_csvs/{model_name}/'
    cre_df_path = f'{csv_dir}/sufficient_CREs.csv'
    cre_set = pd.read_csv(cre_df_path)
    cre_set = cre_set[(cre_set['cell_line'] == cell_line) & (cre_set['tile class'] == 'Enhancer')]



    tile_size = 5000
    # load all motifs from meme file
    motif_file = MotifFile(meme_path)
    motifs = []
    for motif in motif_file:
        motifs.append(motif)

    fimo = FIMO(both_strands=True)

    for r, row in tqdm(cre_set.iterrows(), total=cre_set.shape[0]):
        chrom, start, strand = row['seq_id'].split('_')[1:]

        seq_wt = seq_parser.extract_seq_centered(chrom, int(start), strand, model.seq_length,
                                                 onehot=False).upper()  # wt str whole seq
        seq_onehot_wt = utils.one_hot_encode(seq_wt)  # wt onehot whole seq
        cre_start, cre_end = row['tile_start'], row['tile_end']  # coordinate of tile start
        cre_wt = seq_wt[cre_start:cre_end]  # wt CRE str seq
        cre_onehot_wt = (seq_onehot_wt[cre_start:cre_end]).copy()  # wt CRE onehot

        fasta_path = f'{result_dir}/{row["seq_id"]}_{cre_start}_{cre_end}.fa'  # one seq fasta file
        result_path = f'{result_dir}/{row["seq_id"]}_{cre_start}_{cre_end}.pickle'
        motif_search_res = utils.read_pickle(
            f'../results/motifs_500,50_batch_1,10_shuffle_10_thresh_0.9,0.7/{row["cell_line"]}/{row["seq_id"]}_{cre_start}_{cre_end}.pickle')
        if not os.path.isfile(result_path) and 'control_sequences' in motif_search_res.keys():
            print(result_path)
            with open(fasta_path, 'w') as f:
                f.write(f'>{row["seq_id"]}_{cre_start}_{cre_end} \n')
                f.write(cre_wt)

            sequences = [Sequence(str(record.seq), name=record.id.encode())
                         for record in Bio.SeqIO.parse(fasta_path, "fasta")]  # weird seq format for fimo
            # create N motifs x L mask of motif positions
            mask = np.zeros((len(motifs), tile_size))
            for i, motif in enumerate(motifs):
                pattern = fimo.score_motif(motif, sequences, motif_file.background)
                for m in pattern.matched_elements:
                    mask[i, m.start:m.stop] = 1
            index_bool = [True if m else False for m in np.max(mask, axis=0)]  # mask showing motif positions
            non_index_bool = [not e for e in index_bool]  # mask showing where motifs are not
            random_mask = copy.copy(index_bool)
            np.random.shuffle(random_mask)

            control_sequences = motif_search_res['control_sequences']
            result_summary = {"motif_mask": index_bool}
            for label, mask in {"motifs": index_bool, "non-motifs": non_index_bool, "random": random_mask,
                                "all": [True for _ in range(5000)]}.items():
                # take shuffled CRE from controlseqs
                cre_motifs = control_sequences[:, cre_start: cre_end, :].copy()  # start with shuffled CRE
                cre_motifs[:, mask, :] = cre_onehot_wt[mask]  # embed
                # embed CRE with just the motifs back into seqs
                test_seqs = control_sequences.copy()
                test_seqs[:, cre_start: cre_end, :] = cre_motifs

                test_pred = model.predict(test_seqs)
                result_summary[label] = test_pred.mean()
            utils.save_pickle(result_path, result_summary)


if __name__ == '__main__':
    main()