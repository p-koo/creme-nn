import numpy as np
import pandas as pd
import pyranges as pr
from creme import utils, custom_model 


########################################################################################
# parameters
########################################################################################

# enformer params
SEQUENCE_LEN = 393216
track_index = 5111
bin_index = 448
tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = '../data/hg19.fa'

# file paths
annotation_path = '../data/gencode.v43lift37.basic.annotation.gtf'

# test params
batch_size = 1
thresh = 30
save_path = '../data/TSS.csv'


########################################################################################
# process GENECODE annotations
########################################################################################

# get GENCODE annotations
gencode_annotations = pr.read_gtf(annotation_path)

# subselect anntotations based on feature of interest
feature = 'transcript'
tss_df = gencode_annotations.df.query('Feature=="'+feature+'"')

# get the start locations of annotated genes
tss_positions = []
for i, row in tss_df.iterrows():
    if row['Strand'] == '+':
        tss_positions.append([row['Chromosome'], row['Start'], row['gene_name']])
    elif row['Strand'] == '-':
        tss_positions.append([row['Chromosome'], row['End'], row['gene_name']])
    else:
        print(row)

# create dataframe
tss_df = pd.DataFrame(tss_positions, columns=['chrom', 'tss', 'gene'])
print('Starting with %d genes'%(len(tss_df)))


########################################################################################
# acquire enformer predictions and filter genes-of-interest
########################################################################################

# load enformer model 
model = custom_model.Enformer(tfhub_url, head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

# loop athrough and predict TSS activity
filter_index = []
i = 0
for k, row in tss_df.iterrows():
    if np.mod(i+1, 1000) == 0:
        print("%d out of %d"%(i+1, len(tss_df)))
        filtered_tss_df = tss_df.iloc[filter_index]
        filtered_tss_df.to_csv(save_path)

    # get seequence from reference genome and convert to one-hot
    one_hot = seq_parser.extract_seq_centered(row['chrom'], row['tss'], SEQUENCE_LEN, onehot=True)

    # acquire predictions
    pred = model.predict(one_hot, batch_size=batch_size)

    # check to see if tss has high activity (i.e. larger than thresh) and is the highest signal in prediction
    if pred[0].argmax() == bin_index and pred[0][bin_index] > thresh:
        filter_index.append(i)
    i += 1
filtered_tss_df = tss_df.iloc[filter_index]

print('Total filtered genes: %d'%(len(filtered_tss_df)))


########################################################################################
# filtered out duplicate genes TSS
########################################################################################

filter_index = []
for gene in tss_df['gene'].unique():
    index = tss_df.loc[tss_df['gene'] == gene].index.to_numpy()
    filter_index.append(index[0])
tss_df = tss_df.iloc[filter_index]

# save to file
tss_df.to_csv(save_path)
print('Removed duplicate genes. Filtered genes: %d'%(len(tss_df)))


########################################################################################
# save filtered TSS
########################################################################################

# save the strong tss list to file
filtered_tss_df.to_csv(save_path)




