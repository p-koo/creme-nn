import os
from creme import utils, custom_model 
import kipoiseq
import pandas as pd
import pyranges as pr
from tqdm import tqdm

# This script should be run after analysis_cre_necessity_test.py

# load TSS dataframe (with all TSS positions)
tss_path = '../data/TSS.csv'
tss_df = pd.read_csv(tss_path)

# load results
results_path = '../results/tss_context_dependence_test.pickle'
with open(results_path, 'rb') as fout:
    pred_all = cPickle.load(fout)

# enhancing context
enhancing_index = np.where(pred_all > 0.95)[0]
enhancing_df = tss_df.iloc[filtered_index]
enhancing_df.to_csv('../data/enhancing_context.csv')

# neutral context
neutral_index = np.where(np.abs(pred_all) < 0.2)[0]
neutral_df = tss_df.iloc[neutral_index]
neutral_df.to_csv('../data/neutral_context.csv')

# silencing context
silencing_index = np.where(pred_all < -0.5)[0]
silencing_df = tss_df.iloc[silencing_index]
silencing_df.to_csv('../data/silencing_context.csv')





