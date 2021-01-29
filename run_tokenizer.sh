#!/bin/bash
if [[ "$1" != "" ]]; then
    dir_path="$1"
else
    dir_path=.
fi
for file in findings bg_and_findings impression; do
    lowercase.perl < ${dir_path}/${file} | normalize-punctuation.perl -l en | \
      tokenizer.perl -q -a -l en -threads 4 > ${dir_path}/${file}.norm.tok
done