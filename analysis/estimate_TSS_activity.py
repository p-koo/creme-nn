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



def main():
    model = custom_model.Enformer()
    seq_len = 393216
    data_dir = '../data/'
    results_dir = '../results/'
    fasta_path = f'{data_dir}/hg19.fa'
    tss_csv_path = f'{results_dir}/tss_positions.csv'
    gencode_annotations = pr.read_gtf(f'{data_dir}/gencode.v43lift37.basic.annotation.gtf')
    tss_df = gencode_annotations.df.query('Feature=="transcript"')

    assert len(tss_df['Strand'].unique()) == 2, 'bad strand'
    tss_positions = [row['Start'] if row['Strand']=='+' else row['End'] for _, row in tss_df.iterrows()]
    tss_df['start'] = tss_positions
    tss_df = tss_df[['chromosome', 'start', 'gene', 'Strand']]
    tss_df.to_csv(tss_csv_path)
    print(tss_df.shape)
    seq_parser = utils.SequenceParser(fasta_path)
    for i, row in tqdm(tss_df.iterrows()):
        result_path = f'{results_dir}/{i}.npz'
        if not os.path.isfile(result_path):
            chrom, start = row[:2]
            sequence_one_hot = seq_parser.extract_seq_centered(chrom, start, seq_len)
            wt_pred = model.predict_all(model, sequence_one_hot)
            print(wt_pred.numpy().shape)
            break


if __name__ == '__main__':
    main()
