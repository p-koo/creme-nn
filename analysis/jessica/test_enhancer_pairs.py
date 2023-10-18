import sys
sys.path.append('../../creme/')
import creme
import custom_model
import utils
import numpy as np
import pandas as pd
import shuffle
from tqdm import tqdm
import os





def main():

    model = custom_model.Enformer(head='human', track_index=5111)

    enhancer_data = pd.read_csv('enhancer_pair_coords_200kb.csv')
    fasta_path = '../../data/hg19.fa'
    seq_parser = utils.SequenceParser(fasta_path)

    seq_len = 196608
    output_dir = utils.make_dir('./csvs/')
    enhancer_data = enhancer_data.sample(frac = 1)
    for i, row in tqdm(enhancer_data.iterrows(), total=enhancer_data.shape[0]):
        result_path = f'{output_dir}/{i}.csv'
        print(result_path)
        if not os.path.isfile(result_path):
            chrom, start = row['gene_chrom'], row['gene_start']
            sequence_one_hot = seq_parser.extract_seq_centered(chrom, start, seq_len)
            row['wt'] = np.squeeze(model.predict(sequence_one_hot))[448]
            seq_start_coord = start - seq_len // 2
            relative_start_1, relative_end_1 = row['enhancer_1_start'] - seq_start_coord, row[
                'enhancer_1_stop'] - seq_start_coord
            relative_start_2, relative_end_2 = row['enhancer_2_start'] - seq_start_coord, row[
                'enhancer_2_stop'] - seq_start_coord
            if relative_start_1 >= 0 and relative_start_2 >=0 and relative_end_1 <= seq_len and relative_end_2 <= seq_len:

                for k, tile_set in {
                    'E1': [[relative_start_1, relative_end_1]],
                    'E2': [[relative_start_2, relative_end_2]],
                    'E1&E2': [[relative_start_1, relative_end_1], [relative_start_2, relative_end_2]]
                }.items():
                    seq_mutants = creme.generate_tile_shuffles(sequence_one_hot, tile_set, 10)
                    row[k] = model.predict(seq_mutants)[:, :, 448].mean()
            else:
                print('Bad enhancer')
                for k in preds.keys():
                    row[k] = 'NA'

            pd.DataFrame(row).T.to_csv(result_path)

if __name__ == '__main__':
    main()