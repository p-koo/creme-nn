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
import shuffle
import custom_model
import utils
import glob

model_name = 'enformer'
seq_parser = utils.SequenceParser('../data/GRCh38.primary_assembly.genome.fa')
model = custom_model.Enformer()
df = pd.read_csv('../results/subsample.csv')
result_dir = utils.make_dir('../results/subsample')
bins = [447, 448]
cell_index = 5111
cell_dict = {'K562': 5111, 'GM12878': 5110, 'PC-3': 4824}

df = df.sample(frac=1)

for i, row in tqdm(df.iterrows()):

    cell_index = cell_dict[row['cell line']]
    result_path = f'{result_dir}/{row["cell line"]}_{row["path"].split("/")[-1]}'
    print(result_path)
    if not os.path.isfile(result_path):
        chrom, start, strand = row['path'].split('/')[-1].split('_')[1:4]
        start = int(start)
        strand = strand.split('.')[0]
        seq = seq_parser.extract_seq_centered(chrom, start, strand, model.seq_length)
        mutants = {}
        for sample_size in [10, 25, 50, 100, 200, 500]:
            mutants[sample_size] = []
            for _ in range(sample_size):
                x_mut = shuffle.dinuc_shuffle(seq)
                x_mut[(model.seq_length - 5000) // 2:(model.seq_length + 5000) // 2] = seq[
                                                                                       (model.seq_length - 5000) // 2:(model.seq_length + 5000) // 2].copy()
                pred_mut = model.predict(x_mut)[0, bins, cell_index].mean()

                mutants[sample_size].append(pred_mut)
            mutants[sample_size] = np.array(mutants[sample_size])

        utils.save_pickle(result_path, mutants)