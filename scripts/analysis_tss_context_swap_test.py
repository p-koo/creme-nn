import numpy as np
from six.moves import cPickle
import pandas as pd
from tqdm import tqdm
from creme import utils, custom_model, creme 

########################################################################################
# parameters
########################################################################################

# enformer params
SEQUENCE_LEN = 393216
track_index = 5111
bin_index = 448
tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = '../data/hg19.fa'

# test params
window = 5000
stride = 5000
num_shuffle = 10

# file paths  (enhancing_context.csv, silencing_context.csv, neutral_context.csv)
source_path = '../data/enhancing_context.csv' 
target_path = '../data/neutral_context.csv'
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
    x_target.append(seq_parser.extract_seq_centered(row['chrom'], row['tss'], SEQUENCE_LEN, onehot=True))
x_target = np.array(x_target)

# loop athrough and predict TSS activity
pred_all = []
for i, row in tqdm(tss_df.iterrows()):

    # get seequence from reference genome and convert to one-hot
    x_source = seq_parser.extract_seq_centered(row['chrom'], row['tss'], SEQUENCE_LEN, onehot=True)

    # perform TSS Context Swap Test
    pred_wt, pred_mut = creme.context_swap_test(model, x_source, x_target, tss_tile, mean=True)

    # normalize predictions
    pred_norm = creme.context_effect_on_tss(pred_wt, pred_mut, bin_index)

    # store predictions
    pred_all.append(pred_norm)

# save results
with open(save_path, 'wb') as fout:
    cPickle.dump(np.array(pred_all), fout)






