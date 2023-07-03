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
num_rounds = 25
optimization = np.argmax             # argmin to search for enhancers and argmax to search for silencers
reduce_fun = utils.reduce_pred_index # function to reduce prediction of model to scalar

# file paths
enhancer_path = '../data/enhancers.csv'
save_path = '../results/cre_multiplicity_test.pickle'


########################################################################################
# analysis
########################################################################################


# load TSS dataframe (with all TSS positions)
enhancers_df = pd.read_csv(enhancer_path)

# get coordinates of central tss 
tss_tile, other_tiles = utils.set_tile_range(SEQUENCE_LEN, window, stride)

# load enformer model 
model = custom_model.Enformer(head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

# loop through and predict TSS activity
pred_all = []
positions_all = []
for i, row in tqdm(enhancers_df.iterrows()):

    # get seequence from reference genome and convert to one-hot
    x = seq_parser.extract_seq_centered(row['chrom'], row['tss'], SEQUENCE_LEN, onehot=True)

    # get coordinates for enhancer of interest
    cre_tile = [row['tile_start'], row['tile_start']+window]

    # perform CRE Higher-order Interaction Test
    pred_wt, pred_per_round, max_positions  = creme.multiplicity_test(model, x, 
                                                                      tss_tile, 
                                                                      cre_tile,
                                                                      other_tiles, 
                                                                      num_shuffle, 
                                                                      num_rounds, 
                                                                      optimization,
                                                                      reduce_fun)

    # store predictions
    pred_all.append(pred_per_round/np.mean(pred_control,axis=0)[bin_index])
    positions_all.append(max_positions)

# save results
with open(save_path, 'wb') as fout:
    cPickle.dump(np.array(pred_all), fout)
    cPickle.dump(np.array(positions_all), fout)






