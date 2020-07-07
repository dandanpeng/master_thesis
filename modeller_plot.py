#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:21:03 2020

@author: pengdandan
"""

import pandas as pd
import matplotlib.pyplot as plt


from Bio import Align

reference = 'CASSLLGGWSEAFF' #5tez
reference = 'CASSVAGTPSYEQYF' #5ksa
reference = 'CASSSWDTGELFF' #3vxs
reference = 'CASSIRSSYEQYF' #5euo

def process_data(data_file):
    file = pd.read_csv(data_file, names = ['energy', 'sequence'])
    file['iteration'] = None
    for i in range(1, 21):
        file['iteration'][(i-1)*32:i*32] = i
    return file

cdr3_seq1 = process_data('nt_vae1/cdr3.seq')
cdr3_seq2 = process_data('nt_vae2/cdr3.seq')
cdr3_seq3 = process_data('nt_vae3/cdr3.seq')

cdr3_seq = pd.concat([cdr3_seq1, cdr3_seq2])

cdr3_seq = cdr3_seq[cdr3_seq.iloc[:, 0] < 0]
cdr3_seq_uniq = cdr3_seq.drop_duplicates(subset = 'sequence', keep = 'first')


aligner = Align.PairwiseAligner()

align_score = []
for i in range(len(cdr3_seq_uniq)):
    length = max(len(cdr3_seq_uniq.iloc[i, 1]), len(reference))
    align_score.append(aligner.align(cdr3_seq_uniq.iloc[i, 1], reference).score/length)

cdr3_seq_uniq['identity'] = align_score

fig, ax = plt.subplots()  
cdr3_seq_uniq.plot(kind = 'scatter', x = 'identity', y = 'energy', c = 'iteration', cmap = 'coolwarm', ax = ax, alpha = 0.5)
plt.xlim(0.2, 1)
plt.ylim(-2500, -200)
plt.axhline(y =  -2074.89, color='r', linestyle='-', label = 'original')
plt.axhline(y =  -1668.25, color='b', linestyle='-', label = 'model')
plt.legend()
plt.savefig('/Users/pengdandan/Desktop/spaceml1/TCR_design/dandan_tcr_design/5euo_nt_vae.png', 
            dpi = 300)
