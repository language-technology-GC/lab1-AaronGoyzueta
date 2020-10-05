#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:09:50 2020

@author: aarongoyzueta
"""

import csv
import nltk

from nltk.tokenize import word_tokenize


with open('news.2012.en.shuffled.deduped', 'r') as source:
    with open('data/tokenized_text.tsv', 'w') as sink:
        tsv_output = csv.writer(sink, delimiter='\t')
        line = source.readline().rstrip()
        line_toks = word_tokenize(line)
        tsv_output.writerow(line_toks)
        while line:
            line = source.readline().rstrip()
            line_toks = word_tokenize(line)
            tsv_output.writerow(line_toks)
    