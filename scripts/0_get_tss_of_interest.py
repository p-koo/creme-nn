import os
from creme import utils, custom_model 
import kipoiseq
import pandas as pd
import pyranges as pr
from tqdm import tqdm


########################################################################################
# parameters
########################################################################################

SEQUENCE_LEN = 393216
track_index = 5111
bin_index = 448
batch_size = 1
thresh = 30
save_path = utils.make_dir('../results')
tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
fasta_path = 'hg19.fa'
annotation_path = 'gencode.v43lift37.basic.annotation.gtf'


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
tss_df = pd.DataFrame(tss_positions, columns=['chrom', 'start', 'gene'])


########################################################################################
# acquire enformer predictions and filter genes-of-interest
########################################################################################

# load enformer model 
model = custom_model.Enformer(tfhub_url, head='human', track_index=track_index)

# set up sequence parser from fasta 
seq_parser = utils.SequenceParser(fasta_path)

# loop athrough and predict TSS activity
filtered_index = []
for i, row in tqdm(tss_df.iterrows()):

    # get seequence from reference genome and convert to one-hot
    one_hot = seq_parser.extract_seq_centered(row['chrom'], row['start'], SEQUENCE_LEN, onehot=True)

    # acquire predictions
    pred = model.predict(one_hot, batch_size=batch_size)

    # check to see if tss has high activity (i.e. larger than thresh) and is the highest signal in prediction
    if pred[0].argmax() == bin_index and pred[0][bin_index] > thresh:
        filtered_index.append(i)
filtered_index = np.array(filtered_index)


########################################################################################
# save filtered TSS
########################################################################################

# save the strong tss list to file
tss_df = tss_df.iloc[filtered_index]
tss_df.to_csv(os.path.join(save_path, 'TSS.csv'))







