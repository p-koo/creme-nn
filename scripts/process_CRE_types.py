import numpy as np
import pandas as pd
from six.moves import cPickle

# This script should be run after analysis_cre_sufficiency_test.py


############################################################################################
# Process CRE types
############################################################################################

# parameters
strong_thresh = 0.5
weak_thresh = 0.3

# load TSS dataframe (with all TSS positions)
tss_path = '../data/TSS.csv'
tss_df = pd.read_csv(tss_path)

# CRE files
enhancer_df = pd.DataFrame(columns=['index', 'chrom', 'tss', 'gene', 'context_type', 'tile_start', 'tile_type'])
silencer_df = pd.DataFrame(columns=['index', 'chrom', 'tss', 'gene', 'context_type', 'tile_start', 'tile_type'])

context = 'enhancing'
pickle_path = '../results/cre_sufficiency_test_enhancing_context.pickle'
with open(pickle_path, 'rb') as fout:
    pred_all = cPickle.load(fout)
enhancer_df = parse_enhancers_to_df(tss_df, enhancer_df, pred_all, context, weak_thresh, strong_thresh)
silencer_df = parse_silencers_to_df(tss_df, silencer_df, pred_all, context, weak_thresh, strong_thresh)

context = 'neutral'
pickle_path = '../results/cre_sufficiency_test_neutral_context.pickle'
with open(pickle_path, 'rb') as fout:
    pred_all = cPickle.load(fout)
enhancer_df = parse_enhancers_to_df(tss_df, enhancer_df, pred_all, context, weak_thresh, strong_thresh)
silencer_df = parse_silencers_to_df(tss_df, silencer_df, pred_all, context, weak_thresh, strong_thresh)

context = 'silencing'
pickle_path = '../results/cre_sufficiency_test_silencing_context.pickle'
with open(pickle_path, 'rb') as fout:
    pred_all = cPickle.load(fout)
enhancer_df = parse_enhancers_to_df(tss_df, enhancer_df, pred_all, context, weak_thresh, strong_thresh)
silencer_df = parse_silencers_to_df(tss_df, silencer_df, pred_all, context, weak_thresh, strong_thresh)

# save to file
enhancer_df.to_csv('../data/enhancers.csv')
silencer_df.to_csv('../data/silencers.csv')



############################################################################################
# Useful function
############################################################################################


def parse_enhancers_to_df(tss_df, save_df, pred_all, context, weak_thresh, strong_thresh):

    for j, pred in enumerate(pred_all):
        index = np.where(pred > weak_thresh)[0]
        if np.array(index).any():

            # loop through tiles
            for i in index:

                # add original entry of the sequence information
                vals = []
                for val in tss_df.iloc[j]:
                    vals.append(val)

                # add entry for context type
                vals.append(context)

                # get index of which tiles are strong enhancers
                vals.append(i)
                
                # add label for enhancer strength
                if pred[index] > strong_thresh:
                    vals.append('strong_enhancer')
                else:
                    vals.append('weak_enhancer')

                # add tile information
                save_df.loc[len(save_df.index)] = vals
    return save_df



def parse_silencers_to_df(tss_df, save_df, pred_all, context, weak_thresh, strong_thresh):

    for j, pred in enumerate(pred_all):
        index = np.where(pred < weak_thresh)[0]
        if np.array(index).any():

            # loop through tiles
            for i in index:

                # add original entry of the sequence information
                vals = []
                for val in tss_df.iloc[j]:
                    vals.append(val)

                # add entry for context type
                vals.append(context)

                # get index of which tiles are strong enhancers
                vals.append(i)
                
                # add label for enhancer strength
                if pred[index] < strong_thresh:
                    vals.append('strong_silencer')
                else:
                    vals.append('weak_silencer')

                # add tile information
                save_df.loc[len(save_df.index)] = vals
    return save_df


