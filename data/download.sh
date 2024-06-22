#!/bin/bash

if ! test -f ./GRCh38.primary_assembly.genome.fa; then
  wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz
  gunzip GRCh38.primary_assembly.genome.fa.gz
fi
# wget hg19.fa


if ! test -f ./gencode.v44.basic.annotation.gtf; then
  # wget gencode
  wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz
  gunzip gencode.v44.basic.annotation.gtf.gz
fi

if ! test -f ./enformer_targets_human.txt; then
wget https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt -O enformer_targets_human.txt
fi

if ! test -f ./borzoi_targets_human.txt; then
wget https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt -O borzoi_targets_human.txt
fi

if ! test -f ./enhancer_atlas_K562.bed; then
wget http://www.enhanceratlas.org/data/download/enhancer/hs/K562.bed -O enhancer_atlas_K562.bed
fi

if ! test -f ./borzoi_params_pred.json; then
wget https://raw.githubusercontent.com/calico/borzoi/main/examples/params_pred.json -O borzoi_params_pred.json
fi

#Download model weights
mkdir -p borzoi
for fold in f0 f1 f2 f3; do
  mkdir -p "borzoi/$fold/"
  local_model="borzoi/$fold/model0_best.h5"
  if [ -f "$local_model" ]; then
    echo "$fold model already exists."
  else
    wget --progress=bar:force "https://storage.googleapis.com/seqnn-share/borzoi/$fold/model0_best.h5" -O "$local_model"
  fi
done


