#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:29:48 2020

@author: pengdandan
"""
import click
import collections
import numpy as np
import pandas as pd
import random

from vampire import gene_name_conversion as conversion
from Bio.Seq import Seq

NT_ORDER = 'ACGT-'
NT_LIST = list(NT_ORDER)
NT_DICT = {c: i for i, c in enumerate(NT_LIST)}
NT_DICT_REV = {i:c for i, c in enumerate(NT_LIST)}
NT_SET = set(NT_LIST)
NT_NONGAP = [float(c != '-') for c in NT_LIST]

# Sometimes Adaptive uses one set of column names, and sometimes another.
HEADER_TRANSLATION_DICT = {
    'sequenceStatus': 'frame_type',
    'aminoAcid': 'amino_acid',
    'vGeneName': 'v_gene',
    'jGeneName': 'j_gene',
    'nucleotide': 'rearrangement'
}


def filter_and_drop_frame(df):
    """
    Select in-frame sequences and then drop that column.
    """
    return df.query('frame_type == "In"').drop('frame_type', axis=1)


def filter_on_cdr3_bounding_aas(df):
    """
    Only take sequences that have a C at the beginning and a F or a YV at the
    end of the `amino_acid` column.

    Note that according to the Adaptive docs the `amino_acid` column is indeed
    the CDR3 amino acid.
    """
    return df[df['amino_acid'].str.contains('^C.*F$') | df['amino_acid'].str.contains('^C.*YV$')]


def filter_on_cdr3_length(df, max_len):
    """
    Only take sequences that have a CDR3 of at most `max_len` length.
    """
    return df[df['amino_acid'].apply(len) <= max_len]


def filter_on_TCRB(df):
    """
    Only take sequences that have a resolved TCRB gene for V and J.
    """
    return df[df['v_gene'].str.contains('^TCRB') & df['j_gene'].str.contains('^TCRB')]


def filter_on_olga(df):
    """
    Only take sequences with genes that are present in both the OLGA and the
    Adaptive gene sets.

    Also,
    * exclude TCRBJ2-5, which Adaptive annotates badly.
    * exclude TCRBJ2-7, which appears to be problematic for OLGA.
    """
    d = conversion.adaptive_to_olga_dict()
    del d['TRBJ']['TCRBJ02-05']
    del d['TRBJ']['TCRBJ02-07']
    return conversion.filter_by_gene_names(df, d)


def apply_all_filters(df, max_len=30, fail_fraction_remaining=None):
    """
    Apply all filters.

    Fail if less than `fail_fraction_remaining` of the sequences remain.
    """
    click.echo(f"Original data: {len(df)} rows")
    df = filter_and_drop_frame(df)
    original_count = len(df)
    click.echo(f"Restricting to in-frame: {len(df)} rows")
    df = filter_on_cdr3_bounding_aas(df)
    click.echo(f"Requiring sane CDR3 bounding AAs: {len(df)} rows")
    df = filter_on_cdr3_length(df, max_len)
    click.echo(f"Requiring CDR3 to be <= {max_len} amino acids: {len(df)} rows")
    df = filter_on_TCRB(df)
    click.echo(f"Requiring resolved TCRB genes: {len(df)} rows")
    df = filter_on_olga(df)
    click.echo(f"Requiring genes that are also present in the OLGA set: {len(df)} rows")
    if fail_fraction_remaining:
        if len(df) / original_count < fail_fraction_remaining:
            raise Exception(f"We started with {original_count} sequences and now we have {len(df)}. Failing.")
    return df.reset_index(drop=True)


def collect_vjcdr3_duplicates(df):
    """
    Define a vjcdr3 to be the concatenation of the V label,
    the J label, and the CDR3 protein sequence. Here we build
    a dictionary mapping vjcdr3 sequences to rows containing
    that vjcdr3 sequence.

    We only include sequences with a CDR3 amino acid sequence.
    """
    d = collections.defaultdict(list)

    for idx, row in df.iterrows():
        # nan means no CDR3 sequence. We don't want to include those.
        if row['amino_acid'] is not np.nan:
            key = '_'.join([row['v_gene'], row['j_gene'], row['amino_acid']])
            d[key].append(idx)

    return d


def dedup_on_vjcdr3(df):
    """
    Given a data frame of sequences, sample one
    representative per vjcdr3 uniformly.

    Note: not used in the current preprocessing step.
    """
    dup_dict = collect_vjcdr3_duplicates(df)
    c = collections.Counter([len(v) for (_, v) in dup_dict.items()])
    click.echo("A count of the frequency of vjcdr3 duplicates:")
    click.echo(c)
    indices = [random.choice(v) for (_, v) in dup_dict.items()]
    indices.sort()
    return df.loc[indices].reset_index(drop=True)


def read_adaptive_tsv(path):
    """
    Read an Adaptive TSV file and extract the columns we use, namely
    amino_acid, frame_type, v_gene, and j_gene.

    I have seen two flavors of the Adaptive header names, one of which uses
    snake_case and the other that uses camelCase.
    """
    test_bite = pd.read_csv(path, delimiter='\t', nrows=1)

    camel_columns = set(HEADER_TRANSLATION_DICT.keys())
    snake_columns = set(HEADER_TRANSLATION_DICT.values())

    if camel_columns.issubset(set(test_bite.columns)):
        take_columns = camel_columns
    elif snake_columns.issubset(set(test_bite.columns)):
        take_columns = snake_columns
    else:
        raise Exception("Unknown column names!")

    df = pd.read_csv(path, delimiter='\t', usecols=take_columns)
    df.rename(columns=HEADER_TRANSLATION_DICT, inplace=True)
    return df

def trunc_dna(data):
    data['trunc_rearrangement'] =np.nan
    if 'nucleotide' in data.columns:   
        for i in range(len(data)):
            start = Seq(data['nucleotide'][i]).translate().find(data['aminoAcid'][i])
            end = (start + len(data['aminoAcid'][i]))        
            data['trunc_rearrangement'][i] = data['rearrangement'][i][start*3:end*3] 
    elif 'rearrangement' in data.columns:
        for i in range(len(data)):
            start = Seq(data['rearrangement'][i]).translate().find(data['amino_acid'][i])
            end = (start + len(data['amino_acid'][i]))
            data['trunc_rearrangement'][i] = data['rearrangement'][i][start*3:end*3] 
    data = data.drop('rearrangement', 1)
    data.to_csv(args.output, index = False)
    
def ntseq_to_onehot(codon):
    v = np.zeros((90, 5))
    for i, a in enumerate(codon):
        v[i][NT_DICT[a]] = 1
    return v

def pad(seq, desired_length = 90):
    seq_len = len(seq)
    assert seq_len <= desired_length
    pad_len = desired_length - seq_len
    pad_start = seq_len // 2
    if pad_start % 3 == 0:
        return seq[:pad_start] + '-' * pad_len + seq[pad_start:]
    elif pad_start % 3 == 1:
        return seq[:pad_start-1] + '-' * pad_len + seq[pad_start-1:]
    elif pad_start % 3 == 2:
        return seq[:pad_start+1] + '-' * pad_len + seq[pad_start+1:]

def unpadded_to_onehot(df, desired_length):
    return df['trunc_rearrangement'].apply(lambda s: ntseq_to_onehot(pad(s, desired_length)))

#df = apply_all_filters(read_adaptive_tsv(args.input), fail_fraction_remaining=0.25)
#trunc_dna(df)

