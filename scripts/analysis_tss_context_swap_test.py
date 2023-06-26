import os
import numpy as np
from six.moves import cPickle
from creme import shuffle
import pandas as pd
import pyranges as pr
from tqdm import tqdm
from creme import utils, custom_model 

########################################################################################
# parameters
########################################################################################

SEQUENCE_LEN = 393216
track_index = 5111
bin_index = 448
num_shuffle = 10
window = 5000
stride = 5000
tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = 'hg19.fa'


source_path = 'TSS_enhancing.csv'
target_path = 'TSS_neutral.csv'
save_path = '../results/tss_context_swap_test.pickle'


########################################################################################
# analysis
########################################################################################


# load TSS dataframe (with all TSS positions)
source_df = pd.read_csv(source_path)
target_df = pd.read_csv(target_path)

# get coordinates of central tss 
tss_tile,_ = utils.set_tile_range(SEQUENCE_LEN, window, stride)

# load enformer model 
model = custom_model.Enformer(tfhub_url, head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

x_target = []
for row in tss_df.iterrows():
    # get seequence from reference genome and convert to one-hot
    x_target.append(seq_parser.extract_seq_centered(row['chrom'], row['start'], SEQUENCE_LEN, onehot=True))
x_target = np.array(x_target)

# loop athrough and predict TSS activity
pred_all = []
for i, row in tqdm(tss_df.iterrows()):

    # get seequence from reference genome and convert to one-hot
    x_source = seq_parser.extract_seq_centered(row['chrom'], row['start'], SEQUENCE_LEN, onehot=True)

    # perform TSS Context Swap Test
    pred_wt, pred_mut = perturb.context_swap_test(model, x_source, x_target, tss_tile, mean=True)

    # normalize predictions
    pred_norm = perturb.context_effect_on_tss(pred_wt, pred_mut, bin_index)

    # store predictions
    pred_all.append(pred_norm)

# save results
with open(save_path, 'wb') as fout:
    cPickle.dump(np.array(pred_all), fout)






