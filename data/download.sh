#!/bin/bash

if ! test -f ./hg19.fa; then
  wget https://www.dropbox.com/s/mojkzvlmsw2bk6b/hg19.fa
fi
# wget hg19.fa


if ! test -f ./gencode.v43lift37.basic.annotation.gtf; then
  # wget gencode
  wget https://www.dropbox.com/s/1j8imw5eykjyzvm/gencode.v43lift37.basic.annotation.gtf.gz
  gunzip gencode.v43lift37.basic.annotation.gtf.gz
fi

if ! test -f ./enformer_targets_human.txt; then
wget https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt -O enformer_targets_human.txt
fi

if ! test -f ./borzoi_targets_human.txt; then
wget https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt -O borzoi_targets_human.txt
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


wget https://raw.githubusercontent.com/calico/borzoi/main/examples/params_pred.json -O borzoi_params_pred.json
