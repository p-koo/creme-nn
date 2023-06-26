import os
import numpy as np
from creme import shuffle
import pandas as pd
import pyranges as pr
from tqdm import tqdm
from creme import utils, custom_model 


def context_dependence_test(model, x, tile_pos, num_shuffle, mean=True):
    """
    This test places a sequence pattern bounded by start and end in shuffled 
    background contexts -- in line with a global importance analysis. 

    inputs:
        model: keras model 
        x: a single one-hot sequence shape (L, A)
        tile_pos: list with start index and end index of pattern along L
        num_shuffle: number of shuffles to apply
    """

    # get wild-type prediction
    pred_wt = model.predict(x[np.newaxis])

    # crop pattern of interest
    start, end = tile_pos
    x_pattern = x[start:end,:]

    # loop over shuffles
    pred_mut = []
    for n in range(num_shuffle):
        x_mut = shuffle.dinuc_shuffle(x)
        x_mut[start:end,:] = x_pattern
        pred_mut.append(model.predict(x_mut)[np.newaxis])
    pred_mut = np.array(pred_mut)

    if mean:
        return np.mean(pred_wt, axis=0), np.mean(pred_mut, axis=0)
    else:
        return pred_wt, pred_mut 

########################################################################################
# parameters
########################################################################################

SEQUENCE_LEN = 393216
track_index = 5111
bin_index = 448
num_shuffle = 10
batch_size = 1
window = 5000
stride = 5000
save_path = utils.make_dir('../results')
tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = 'hg19.fa'
tss_path = 'TSS.csv'


# create dataframe
tss_df = pd.read_csv(tss_path)

# get coordinates of central tss 
tss_tile, other_tiles = utils.set_tile_range(SEQUENCE_LEN, window, stride)


########################################################################################
# acquire enformer predictions and filter genes-of-interest
########################################################################################

# load enformer model 
model = custom_model.Enformer(tfhub_url, head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

# loop athrough and predict TSS activity
pred_all = []
for i, row in tqdm(tss_df.iterrows()):

    # get seequence from reference genome and convert to one-hot
    one_hot = seq_parser.extract_seq_centered(row['chrom'], row['start'], SEQUENCE_LEN, onehot=True)

    pred_wt, pred_mut = perturb.context_dependence_test(model, x, tss_tile, num_shuffle, mean=True)

    pred_norm = context_effect_on_tss(pred_wt, pred_mut, bin_index)

    pred_all.append(pred_norm)








