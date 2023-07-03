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
fasta_path = '../data/hg19.fa'

# test params
window = 5000
stride = 5000
num_shuffle = 10

# file paths
tss_path = '../data/enhancing_context.csv'  # silencing_context.csv, neutral_context.csv
save_path = '../results/cre_necessity_test_enhancing_context.pickle'


########################################################################################
# analysis
########################################################################################


# load TSS dataframe (with all TSS positions)
tss_df = pd.read_csv(tss_path)

# get coordinates of central tss 
tss_tile,other_tiles = utils.set_tile_range(SEQUENCE_LEN, window, stride)

# load enformer model 
model = custom_model.Enformer(head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

# loop athrough and predict TSS activity
pred_all = []
for i, row in tqdm(tss_df.iterrows(), total=len(tss_df)):

    # get seequence from reference genome and convert to one-hot
    x = seq_parser.extract_seq_centered(row['chrom'], row['tss'], SEQUENCE_LEN, onehot=True)

    # perform CRE Necessity Test
    pred_wt, pred_mut = creme.necessity_test(model, x, other_tiles, num_shuffle, mean=True)

    # normalize predictions
    pred_norm = creme.context_effect_on_tss(pred_wt, pred_mut, bin_index)

    # store predictions
    pred_all.append(pred_norm)

# save results
with open(save_path, 'wb') as fout:
    cPickle.dump(np.array(pred_all), fout)






