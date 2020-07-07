#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:39:50 2020

@author: pengdandan
"""

import glob
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats

plt.style.use("ggplot")

keck = glob.glob('Keck*.pvae.csv')
olga_keck = glob.glob('../../../../../../vampire/vampire/pipe_main/_output_emerson-2017-03-04/Keck*')

vae_seq = pd.read_csv('trained_generated.pvae.csv')
svae = pd.read_csv('0.6_generated.pvae.csv')
rnn = pd.read_csv('rnn_generated.pvae.csv')
rnn_by_olga = pd.read_csv('0.6_rnn_generated.ppost.csv', header = None)

distance = []
for i in keck:
    data = pd.read_csv(i)
    distance.append(stats.ks_2samp(data['log_p_x'], rnn['log_p_x'])[0])

np.mean(distance)


olga_keck = glob.glob('../../../../../../vampire/vampire/pipe_main/_output_emerson-2017-03-04/Keck*')
rnn_by_olga = pd.read_csv('0.6_rnn_generated.ppost.csv', header = None)

distance = []
for i in olga_keck:
    data = pd.read_csv(i + '/test-head.pgen.tsv', sep = '\t', header = None)
    distance.append(stats.ks_2samp(data.iloc[:, 3], rnn_by_olga.iloc[:, 1])[0])
np.mean(distance)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

distance_matrix = np.array([[0.12, 0.19, 0.14, 0.28],
                    [0.17, 0.14, 0.04, 0.14],
                    [0.18, 0.0, 0.08, 0.0],
                    [0.16, 0.20, 0.08, 0.03]])
generator = ['OLGA(.Q)', 'VAE-seq', 'SVAE', 'RNN-VAE']

fig, ax = plt.subplots()

im, cbar = heatmap(distance_matrix, generator, generator, ax=ax,
                   cmap="YlGn", cbarlabel="maximum distance between ECDFs")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.savefig('heatmap.png', dpi = 300)

fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(vae_seq['log_p_x'], svae['log_p_x'],  alpha = 0.09, c = '#ff0000')
axs[0, 0].set(xlabel = 'VAE-seq', ylabel = 'SVAE')
axs[0, 1].scatter(vae_seq['log_p_x'], rnn_vae['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[0, 1].set(xlabel = 'VAE-seq', ylabel = 'RNN-VAE')
axs[0, 2].scatter(vae_seq['log_p_x'], nt_vae['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[0, 2].set(xlabel = 'VAE-seq', ylabel = 'nt-VAE')
axs[1, 0].scatter(svae['log_p_x'], rnn_vae['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[1, 0].set(xlabel = 'SVAE', ylabel = 'RNN-VAE')
axs[1, 1].scatter(svae['log_p_x'], nt_vae['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[1, 1].set(xlabel = 'SVAE', ylabel = 'nt-VAE')
axs[1, 2].scatter(rnn_vae['log_p_x'], nt_vae['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[1, 2].set(xlabel = 'RNN-VAE', ylabel = 'nt-VAE')

custom_xlim = (-60, 0)
custom_ylim = (-60, 0)
plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
fig.tight_layout()

plt.savefig('/Users/pengdandan/Desktop/spaceml1/pengd/thesis/data/thesis data/freq_compare.png', dpi =300)



fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(vae_seq['log_p_x'], olga['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[0, 0].set(xlabel = 'VAE-seq', ylabel = 'OLGA')
axs[0, 1].scatter(svae['log_p_x'], olga['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[0, 1].set(xlabel = 'SVAE', ylabel = 'OLGA')
axs[0, 2].scatter(rnn_vae['log_p_x'], olga['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[0, 2].set(xlabel = 'RNN-VAE', ylabel = 'OLGA')
axs[1, 0].scatter(nt_vae['log_p_x'], olga['log_p_x'], alpha = 0.09, c = '#ff0000')
axs[1, 0].set(xlabel = 'nt-VAE', ylabel = 'OLGA')
custom_xlim = (-60, 0)
custom_ylim = (-60, 0)
plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
fig.tight_layout()
plt.savefig('/Users/pengdandan/Desktop/spaceml1/pengd/thesis/data/thesis data/freq_olga_compare.png', dpi =300)
