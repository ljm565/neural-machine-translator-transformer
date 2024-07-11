#!/bin/sh

## setup
dpath=${1}
tpath=${2}
vocab_size=${3}

echo "Data path: $dpath"
echo "Tokenizers path: $tpath"
echo "Vocab size: $vocab_size"

## cat both en and de data
cat $dpath/train.en $dpath/val.en $dpath/test.en $dpath/train.de $dpath/val.de $dpath/test.de | shuf > $dpath/train.all.both

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 ${4} --data $dpath/train.all.both --size $vocab_size --output $tpath/vocab_$vocab_size