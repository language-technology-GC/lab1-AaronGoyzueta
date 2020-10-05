#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:42:42 2020

@author: aarongoyzueta
"""

import argparse
import numpy as np
import pandas as pd

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from scipy import stats

brown_ic = wordnet_ic.ic("ic-brown.dat")

def path_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            temp = x_word.path_similarity(y_word)
            if temp:
                if temp > sim:
                    sim = temp
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def resnik_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            try:
                temp = x_word.res_similarity(y_word, brown_ic)
                if temp:
                    if temp > sim:
                        sim = temp
            except:
                pass
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def wup_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            temp = x_word.wup_similarity(y_word)
            if temp:
                if temp > sim:
                    sim = temp
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def lch_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            try:
                temp = x_word.lch_similarity(y_word)
                if temp:
                    if temp > sim:
                        sim = temp
            except:
                pass
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def jcn_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            try:
                temp = x_word.jcn_similarity(y_word, brown_ic)
                if temp:
                    if temp > sim:
                        sim = temp
            except:
                pass
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def lin_similarity(x, y):
    sim = 0
    for x_word in x:
        for y_word in y:
            try:
                temp = x_word.lin_similarity(y_word, brown_ic)
                if temp:
                    if temp > sim:
                        sim = temp
            except:
                pass
    if sim == 0:
        return None
    else:   
        return round(sim, 2)

def coverage(col1, col2):
    uncovered = col2.isna().sum()
    total = len(col2)
    covered = total - uncovered
    return covered

def correlation(col1, col2):
    return round(stats.spearmanr(col1, col2)[0], 4)


def main(args: argparse.Namespace) -> None:
    total = 0
    covered = 0
    with open("data/final_results.txt", 'a') as sink:
        with open("data/ws353.tsv") as source1:
            if args.target_tsv:
                with open(args.target_tsv) as source2:
                    d2 = {}
                    for line in source2:
                        word1, word2, value = line.rstrip().split('\t')
                        d2[f"{word1}, {word2}"] = value
                    results = {}
                    for line in source1:
                        word1, word2, gold = line.rstrip().split('\t')
                        try:
                            value = d2[f"{word1}, {word2}"]
                            results[f"{word1}, {word2}"] = (gold, value)
                            covered += 1
                        except:
                            results[f"{word1}, {word2}"] = (gold, None)
                        total += 1
                df = pd.DataFrame.from_dict(results, orient='index')
                df = df.fillna(value=np.nan)
                cov = coverage(df.iloc[:, 0], df.iloc[:, 1])
                df = df.dropna()
                corr = correlation(df.iloc[:, 0], df.iloc[:, 1])
                print(f"Method: {args.target_tsv}", file=sink)
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print("\n", file=sink)
            else:
                d = {}
                for line in source1:
                    word1, word2, gold = line.rstrip().split('\t')
                    pair = f"{word1}, {word2}"
                    d[pair] = {}
                    d[pair]["gold"] = gold
                    w1 = wn.synsets(word1)
                    w2 = wn.synsets(word2)
                    d[pair]["path"] = path_similarity(w1, w2)
                    d[pair]["resnik"] = resnik_similarity(w1, w2)
                    d[pair]["wup"] = wup_similarity(w1, w2)
                    d[pair]["lch"] = lch_similarity(w1, w2)
                    d[pair]["jcn"] = jcn_similarity(w1, w2)
                    d[pair]["lin"] = lin_similarity(w1, w2)
                df = pd.DataFrame.from_dict(d, orient='index')
                print("Method: PATH", file=sink)
                corr = correlation(df['gold'], df['path'])
                cov = coverage(df['gold'], df['path'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)
                print("Method: RESNIK", file=sink)
                corr = correlation(df['gold'], df['resnik'])
                cov = coverage(df['gold'], df['resnik'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)
                print("Method: WUP", file=sink)
                corr = correlation(df['gold'], df['wup'])
                cov = coverage(df['gold'], df['wup'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)
                print("Method: LCH", file=sink)
                corr = correlation(df['gold'], df['lch'])
                cov = coverage(df['gold'], df['lch'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)
                print("Method: JCN", file=sink)
                corr = correlation(df['gold'], df['jcn'])
                cov = coverage(df['gold'], df['jcn'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)
                print("Method: LIN", file=sink)
                corr = correlation(df['gold'], df['lin'])
                cov = coverage(df['gold'], df['lin'])
                print(f"Correlation: {corr}", file=sink)
                print(f"Coverage: {cov}", file=sink)
                print('\n', file=sink)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_tsv",
        default=None,
        help="path to input tsv to compare to human judgements data"
    )
    main(parser.parse_args())