import os
from creme import utils, custom_model 
import kipoiseq
import pandas as pd
import pyranges as pr


SEQUENCE_LENGTH = 393216
PADDING = 196608

track_index = 5111
bin_index = 448
num_shuffle = 10
batch_size = 1
save_path = '../results'
out_dir = utils.make_dir(save_path)

tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = '/home/shush/genomes/hg19.fa'


########################################################################################
# process GENECODE annotations
########################################################################################


# get GENCODE annotations
annotation_path = 'gencode.v43lift37.basic.annotation.gtf'
gencode_annotations = pr.read_gtf(annotation_path)

# subselect anntotations based on feature of interest
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
tss_df = pd.DataFrame(tss_positions, columns=['chromosome', 'start', 'gene'])


########################################################################################
# acquire enformer predictions and filter genes-of-interest
########################################################################################


# load enformer model and fasta-extractor
model = custom_model.Enformer(tfhub_url, head='human', track_index=track_index)
fasta_extractor = utils.FastaStringExtractor(fasta_path)

# loop through and predict TSS activity
filtered_index = []
for i, row in tss_df.iterrows():

    # get tss location
    tss = row['start']

    # get coordinates for tss
    target_interval = kipoiseq.Interval(row['chromosome'], tss, tss + 1).resize(SEQUENCE_LENGTH)

    # get seequence from reference genome
    seq = fasta_extractor.extract(target_interval)
    
    # convert to one-hot
    sequence_one_hot = utils.one_hot_encode(seq)

    # acquire predictions
    pred = model.predict(sequence_one_hot, batch_size=batch_size)

    # check to see if tss has high activity (i.e. larger than thresh) and is the highest signal in prediction
    if pred[0].argmax() == bin_index and pred[0][bin_index] > thresh:
        filtered_index.append(i)


########################################################################################
# save filtered TSS
########################################################################################

# save the strong tss list to file
tss_df = tss_df[filtered_index]
tss_df.to_csv(os.path.join(save_path, 'TSS.csv'))







