# CREME manuscript analysis
This document walks through the commands for reproducing the analysis in the manuscript. 
## Enformer
For Enformer analysis the following steps were performed. Unless stated otherwise all the csvs are saved
in `../results/summary_csvs/enformer` and all the results are saved as subdirectories in `../results/`. 

### 1.Filter TSS positions with high activity 
We obtained TSS predictions for GENCODE annotations of transcription starts of protein-coding genes.
    ```
    ./estimate_TSS_activity.py enformer
    ./filter_tss.py enformer
    ```
    The first command generates a csv file `../results/tss_positions.csv` and predictions for 
    each of the TSS positions in that file. The second command filters top 10,000 TSS positions
    of unique genes per cell line and generate `*_selected_genes.csv` where * is the cell line
    name.


### 2. Context dependence test
Here we estimated context dependence of 10,000 genes (shuffling N=100 times) 
and select enhancing, silencing and neutral context sequences.
    ```
    ./context_dependence_test.py enformer 100
    ```
    This outputs a csv file `*_selected_contexts.csv` that will
    be used in subsequent steps.


### 3. Context swap test
We embedded each TSS from the categorized sequences from step 2 into all the others. 
    Note, this step uses `*_selected_contexts.csv` from step 2 as a starting
    point to select sequences and generates `context_swap_test.csv`
```
context_swap.py enformer
```
  

### 4. Necessity test
Here we took the categorized sequences from step 2, shuffled non-overlapping 5Kb tiles
   (other than the central TSS tile). The output is a summary table of the results per tile in 
    `necessity_test.csv` and the selected necessary CRE list in `necessary_CREs.csv`. 
   ```
    necessity_test.py enformer
   ```

### 5. Sufficiency test
This test takes the categorized sequences from step 2, shuffles to generate backgrounds and 
    embeds non-overlapping 5Kb tiles into backgrounds (together with the central TSS tile). The output is a 
    summary table of the results per tile in 
        `sufficiency_test.csv` and the selected necessary CRE list in `sufficient_CREs.csv`. 
   ```
    sufficiency_test.py enformer 10
   ```
   
### 6. Biochemical marks of sufficient CREs
For this, we first downloaded the relevant bigwigs:
```
./download_epigenetic_marks.py GM12878,PC-3,K562
```
This will output the marks in `../data/biochemical_marks/*` where * is each cell line.  
Using these we'll process the bigwigs to get the coverage values for the sufficient CREs (based on 
the summary csv `sufficient_CREs.csv`):
```
./process_biochemical_marks.py
```
This creates a directory `../results/biochemical_marks` and saves csv files with mean or max coverage of
each kind of biochemical mark and sufficient CRE.


### 7. RepeatMasker analysis 
For sufficient enhancing and silencing CREs we estimated the number of repetitive
elements in each cell type. This is done in the notebook `figures/RepeatMasker.ipynb` and generates the fasta
files of the sequences in each CRE type. After this we ran RepeatMasker using the defaults:
```
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/GM12878_enhancers.fa
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/GM12878_silencers.fa
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/PC-3_enhancers.fa
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/PC-3_silencers.fa
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/K562_enhancers.fa
RepeatMasker --species human -dir ../results/repeatmasker ../results/XSTREME/K562_silencers.fa
```
This outputs several files including a tabular data with extension .tbl which was used in the notebook to 
extract the relevant information.


### 8. Fine-tile search
Here we have compared CREME's fine-tile search with XSTREME and saliency-based fine-tile
search. This is done through 3 separate scripts and an additional script that generates the summary table of results.
For CREME's test run:
```
./motif_pruning.py 500,50 0.9,0.7 1,10
```
This will run a 2-stage motif pruning, first with window size of 500bp, threshold of 0.9 and one tile at
a time. Then in the 2nd stage it will switch to pruning 50bp tiles, 10 at a time and threshold of 0.7 for
stopping.

For XSTREME we first ran the meme file generation online at https://meme-suite.org/meme/tools/xstreme using enhancing and silencing CRE sequence fasta files
generated in step 7. This created a meme file per cell type for enhancing CREs 
(e.g. `../results/XSTREME/K562_enhancers.meme`). 

Afterwards, we run the script to embed XSTREME motifs in the same backgrounds as fine-tile search:
```
./XSTREME.py ../results/XSTREME/K562_enhancers.meme 5111
./XSTREME.py ../results/XSTREME/GM12878_enhancers.meme 5110
./XSTREME.py ../results/XSTREME/PC-3_enhancers.meme 4824
```
This outputs a result directories at `../results/XSTREME/FIMO`.

Thirdly, we did the saliency-based fine-tile embedding. TO do this we ran:
```
./saliency_analysis.py 5111
./saliency_analysis.py 5110
./saliency_analysis.py 4824
```
which outputs results in `../results/saliency/`.

Finally, we combined all the results by running:
```
./process_motifs.py
```
This outputs `../results/summary_csvs/enformer/motif_analysis/CREME_vs_saliency_vs_XSTREME.csv` and
examples of sequences in `../results/summary_csvs/enformer/motif_analysis/example_seqs.csv`.


### 9. Distance test 
For sufficient CREs in `sufficient_CREs.csv` we ran:
```
./distance_test.py enformer 10 True
```
This generates the directory `../results/distance_test_True/enformer_10/` and the summary csv file
`distance_test.csv`.

### 10. Higher-order interaction test
For this test we ran two commands - one for enhancer and the other for silencer set identification.
```
./higher_order_test.py enformer min
./higher_order_test.py enformer max
```
which output results in `../results/higher_order_test_min` and `../results/higher_order_test_max` as well as 
summary of results in `../results/summary_csvs/enformer/greedy_search/`. This contains results for both runs 
(distinguished) by file prefixes, including `*_second_iteration.csv`, `*_traces.csv` and `*_locations.csv` which are 
required for generating the box plots comparing second iteration results, the summary traces as well as the
location heatmaps.

### 11. CRE set sufficiency
Complementary to step 10 we also embedded the identified CREs into randomized backgrounds to estimate their 
sufficiency. 
```
./sufficiency_of_greedy_search.py enformer min
./sufficiency_of_greedy_search.py enformer max
```

This outputs results in `../results/higher_order_test_min/enformer/sufficiency/` and 
`../results/higher_order_test_max/enformer/sufficiency/` and saves the summary of results in 
`greedy_search/sufficiency_of_greedy_tiles_min.csv` and `greedy_search/sufficiency_of_greedy_tiles_max.csv`.

### 12. Multiplicity test
Here we measured saturation of gene expression prediction as more CRE copies are added:
```
./multiplicity_test.py enformer
```
which outputs results in `../results/multiplicity_test/` and summary table in `multiplicity.csv`.


## Borzoi

The results for Borzoi were generated using the same scripts as Enformer by running:

```
./estimate_TSS_activity.py borzoi
./filter_tss.py borzoi
./context_dependence_test.py borzoi 5
./sufficiency_test.py borzoi 5
./distance_test.py borzoi 5 True
```

These generate the same set of files as described for Enformer. The results are summarized using:
```
./process_borzoi_results.py
```