#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:00:25 2020

@author: pengdandan
"""

import time, click
import math
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Activation, Dense, Lambda, Input, Reshape
from keras import backend as K
from keras import objectives

from keras.callbacks import EarlyStopping, ModelCheckpoint

import scipy.special as special
import scipy.stats as stats

import tensorflow as tf
import vampire.xcr_vector_conversion as conversion

from vampire.custom_keras import BetaWarmup, EmbedViaMatrix
from sklearn.decomposition import PCA

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pad', choices = ('middle', 'front', 'end', 'around', 'startandend'))
parser.add_argument('--gpu', type = str)
parser.add_argument('--latent', type = int)
parser.add_argument('--dense_nodes', type = int)
parser.add_argument('--beta', type = float)
parser.add_argument('--train_file', type = str)
parser.add_argument('--model_weights', type = str)
parser.add_argument('--input', type = str)
parser.add_argument('--output', type = str)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess) 

def default_params():
    """
    Return a dictionary with default parameters.

    The parameters below should be self explanatory except for:

    * beta is the weight put on the KL term of the VAE. See the models for
      how it gets incorporated.
    """
    return dict(
        # Models:
        model='basic',
        # Model parameters.
        latent_dim=args.latent,
        dense_nodes=args.dense_nodes,
        aa_embedding_dim=21,
        v_gene_embedding_dim=30,
        j_gene_embedding_dim=13,
        beta=args.beta,
        # Input data parameters.
        max_cdr3_len=30,
        n_aas=len(conversion.AA_LIST),
        n_v_genes=len(conversion.TCRB_V_GENE_LIST),
        n_j_genes=len(conversion.TCRB_J_GENE_LIST),
        # Training parameters.
        stopping_monitor='val_loss',
        batch_size=100,
        pretrains=10,
        warmup_period=20,
        epochs=500,
        patience=20)

params = default_params()

def sampling(args):
    """
    This function draws a sample from the multivariate normal defined by
    the latent variables.
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
    # Reparameterization trick!
    return (z_mean + K.exp(z_log_var / 2) * epsilon)

def create_model(params):  
    
    beta = K.variable(params['beta'])  
    
    # Input:
    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])
    cdr3_input = Input(shape=cdr3_input_shape, name='cdr3_input')
    
    # Encoding layers:
    cdr3_embedding = EmbedViaMatrix(params['aa_embedding_dim'], name='cdr3_embedding')(cdr3_input)
    cdr3_embedding_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='cdr3_embedding_flat')(cdr3_embedding)
    encoder_dense_1 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_1')(cdr3_embedding_flat)
    encoder_dense_2 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_2')(encoder_dense_1)
    
    # Latent layers:
    z_mean = Dense(params['latent_dim'], name='z_mean')(encoder_dense_2)
    z_log_var = Dense(params['latent_dim'], name='z_log_var')(encoder_dense_2)
    
    # Decoding layers:
    z_l = Lambda(sampling, output_shape=(params['latent_dim'], ), name='z')
    decoder_dense_1_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_1')
    decoder_dense_2_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_2')
    cdr3_post_dense_flat_l = Dense(np.array(cdr3_input_shape).prod(), activation='linear', name='cdr3_post_dense_flat')
    cdr3_post_dense_reshape_l = Reshape(cdr3_input_shape, name='cdr3_post_dense')
    cdr3_output_l = Activation(activation='softmax', name='cdr3_output')
    
    post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var])))
    cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(post_decoder)))
    
    # Define the decoder components separately so we can have it as its own model.
    z_mean_input = Input(shape=(params['latent_dim'], ))
    decoder_post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_mean_input))
    decoder_cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(decoder_post_decoder)))
    encoder = Model(cdr3_input, [z_mean, z_log_var])
    decoder = Model(z_mean_input, [decoder_cdr3_output])
    vae = Model(cdr3_input, cdr3_output)
    def vae_cdr3_loss(io_encoder, io_decoder):
        """
        The loss function is the sum of the cross-entropy and KL divergence. KL
        gets a weight of beta.
        """
        # Here we multiply by the number of sites, so that we have a
        # total loss across the sites rather than a mean loss.
        xent_loss = params['max_cdr3_len']* K.mean(objectives.categorical_crossentropy(io_encoder, io_decoder))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss *= K.variable(params['beta'])
        return xent_loss + kl_loss
  
    def reconstruction_loss(io_encoder, io_decoder):
        return params['max_cdr3_len'] * K.mean(objectives.categorical_crossentropy(cdr3_input, cdr3_output))
    def kl_loss(io_encoder, io_decoder):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    def identity(io_encoder, io_decoder):
        encoder_indice = K.argmax(io_encoder)
        decoder_indice = K.argmax(io_decoder)
        mask = K.not_equal(encoder_indice, 20)  
        encoder_indice_wo_gap = tf.boolean_mask(encoder_indice, mask)
        decoder_indice_wo_gap = tf.boolean_mask(decoder_indice, mask)
        return K.cast(K.equal(encoder_indice_wo_gap, decoder_indice_wo_gap), K.floatx())  
    def perplexity(io_encoder, io_decoder):
        return 30*K.sqrt(1/K.prod(K.sum(io_encoder*io_decoder, axis = 1)))
    vae.compile(
        optimizer="adam",
        loss = vae_cdr3_loss,
        metrics = [reconstruction_loss, kl_loss, identity, perplexity]
        )
    callbacks = [BetaWarmup(beta, params['beta'], params['warmup_period'])]
    return vae, encoder, decoder, callbacks
        
def reinitialize_weights():
        session = K.get_session()
        for layer in vae.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
                
def fit(train_file: str, validation_split: float, best_weights_fname: str, tensorboard_log_dir: str):
    """
    Fit the vae with warmup and early stopping.
    """
    train_csv = pd.read_csv(train_file)
    train_data = conversion.unpadded_tcrbs_to_onehot(train_csv, params['max_cdr3_len'], args.pad)
    cdr3 = np.array(train_data.iloc[:, 0].tolist())
    #v_gene = np.array(train_data.iloc[:, 1].tolist())
    #j_gene = np.array(train_data.iloc[:, 2].tolist())
    
    best_val_loss = np.inf

    # We pretrain a given number of times and take the best run for the full train.
    for pretrain_idx in range(params['pretrains']):
        reinitialize_weights()
        # In our first fitting phase we don't apply EarlyStopping so that
        # we get the number of specifed warmup epochs.
        # Below we apply the fact that right now the only thing in self.callbacks is the BetaSchedule callback.
        # If other callbacks appear we'll need to change this.
        if tensorboard_log_dir:
            callbacks = [keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir + '_warmup_' + str(pretrain_idx))]
        else:
            callbacks = []
        callbacks += callbacks  # <- here re callbacks
        history = vae.fit(
            x=cdr3, 
            y = cdr3, # y=X for a VAE.
            epochs=1 + params['warmup_period'],
            batch_size=params['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0)
        new_val_loss = history.history['val_loss'][-1]
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss
            vae.save_weights(best_weights_fname, overwrite=True)
    
    vae.load_weights(best_weights_fname)

    checkpoint = ModelCheckpoint(best_weights_fname, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(
        monitor=params['stopping_monitor'], patience=params['patience'], mode='min')
    callbacks = [checkpoint, early_stopping]
    if tensorboard_log_dir:
        callbacks += [keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)]
    vae.fit(
        x=cdr3,  # y=X tfor a VAE.
        y=cdr3, 
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2)

def identity_score(original, predict):
    encoder_indice = np.argmax(original, axis = 1)
    decoder_indice = np.argmax(predict, axis = 1)
    
    encoder_indice_wo_gap = encoder_indice[encoder_indice != 20]
    decoder_indice_wo_gap = decoder_indice[encoder_indice != 20]
    return np.sum(np.equal(encoder_indice_wo_gap, decoder_indice_wo_gap)) / len(encoder_indice_wo_gap)
    
def evaluate_test_identity(data_filename):
    data_csv = pd.read_csv(data_filename)
    data = conversion.unpadded_tcrbs_to_onehot(data_csv, params['max_cdr3_len'], args.pad)
    cdr3 = np.array(data.iloc[:, 0].tolist())
     
    z_mean, z_log_var = encoder.predict(cdr3)
    z_sd = np.sqrt(np.exp(z_log_var))
    z_sample = stats.norm.rvs(z_mean, z_sd)
    aa_array = decoder.predict(z_sample)
    
    total_identity = 0
    for i in range(len(cdr3)):
        total_identity += identity_score(cdr3[i], aa_array[i])
    print(total_identity/len(cdr3))

def evaluate_test_perplexity(data_filename):
    data_csv = pd.read_csv(data_filename)
    data = conversion.unpadded_tcrbs_to_onehot(data_csv, params['max_cdr3_len'], 'middle')
    cdr3 = np.array(data.iloc[:, 0].tolist())    
    z_mean, z_log_var = encoder.predict(cdr3)
    z_sd = np.sqrt(np.exp(z_log_var))
    z_sample = stats.norm.rvs(z_mean, z_sd)
    aa_array = decoder.predict(z_sample)    
    total_perplexity = 0
    failed = 0
    for i in range(len(aa_array)):
        pred_probability = np.sum(aa_array[i] * cdr3[i], axis=1)
        if np.prod(pred_probability) == 0:
            print(i)
            failed += 1
        else:
            perplexity = 30 * np.sqrt(1/np.prod(pred_probability))
            total_perplexity += perplexity
    return total_perplexity/(10000 - failed)
    
    
def logprob_of_obs_vect(probs, obs):
    return np.sum(np.log(np.sum(probs * obs, axis=1)))

def log_pvae_importance_sample(x_df, out_ps):   
    # We're going to be getting a one-sample estimate, so we want one slot
    # in our output array for each input sequence.
    assert (len(x_df) == len(out_ps))
    # Get encoding of x's in the latent space.
    z_mean, z_log_var = encoder.predict(x_df)
    z_sd = np.sqrt(np.exp(z_log_var))
    # Get samples from q(z|x) in the latent space, one for each input x.
    z_sample = stats.norm.rvs(z_mean, z_sd)
    # These are decoded samples from z. They are, thus, probability vectors
    # that get sampled if we want to realize actual sequences.
    aa_probs = decoder.predict(z_sample)
    
    # Onehot-encoded observations.
    # We use interpret_output to cut down to what we care about.
    aa_obs = x_df
    # Loop over observations.
    for i in range(len(x_df)):
        log_p_x_given_z = \
            logprob_of_obs_vect(aa_probs[i], aa_obs[i])
        # p(z)
        # Here we use that the PDF of a multivariate normal with
        # diagonal covariance is the product of the PDF of the
        # individual normal distributions.
        log_p_z = np.sum(stats.norm.logpdf(z_sample[i], 0, 1))
        # q(z|x)
        log_q_z_given_x = np.sum(stats.norm.logpdf(z_sample[i], z_mean[i], z_sd[i]))
        # Importance weight: p(z)/q(z|x)
        log_imp_weight = log_p_z - log_q_z_given_x
        # p(x|z) p(z) / q(z|x)
        out_ps[i] = log_p_x_given_z + log_imp_weight
        
def pvae(nsamples, test_csv, out_csv):
    df = pd.read_csv(test_csv)
    #df_x = conversion.unpadded_tcrbs_to_onehot(df, params['max_cdr3_len'], args.pad)
    cdr3_series = df['amino_acid'].apply(lambda s: conversion.seq_to_onehot(conversion.pad(s, params['max_cdr3_len'], args.pad)))
    cdr3 = np.array(cdr3_series.tolist())
    #cdr3 = np.array(df_x.iloc[:, 0].tolist())
    #v_gene = np.array(df_x.iloc[:, 1].tolist())
    #j_gene = np.array(df_x.iloc[:, 2].tolist())
    log_p_x = np.zeros((nsamples, len(cdr3)))
    click.echo("Calculating pvae for {} via importance sampling...")    
    with click.progressbar(range(nsamples)) as bar:
        for i in bar:
            log_pvae_importance_sample(cdr3, log_p_x[i])    
    avg = special.logsumexp(log_p_x, axis=0) - np.log(nsamples)
    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index=False)


def add_pcs(df):
    """
    Add principal component information to a copy TCR data frame.
    """
    df_x = pd.read_csv(df)
    cdr3_series = df_x['amino_acid'].apply(lambda s: conversion.seq_to_onehot(conversion.pad(s, params['max_cdr3_len'], args.pad)))
    cdr3 = np.array(cdr3_series.tolist())
    
    z_mean,_ = encoder.predict(cdr3)
    pca = PCA(n_components=2)
    pca.fit(z_mean)
    z_mean_pcs = pca.transform(z_mean)
    df_x = pd.DataFrame(df_x)
    df_x['pc_1'] = z_mean_pcs[:, 0]
    df_x['pc_2'] = z_mean_pcs[:, 1]
    df_x.to_csv('pcs.csv', index=False)

#df_topgenes = df.loc[(df['v_gene'] == 'TCRBV30-01') & (df['j_gene'] == 'TCRBJ01-02'),]
#add_pcs(v, df_topgenes).to_csv('output/pcs_topgenes.csv', index=False)

def generate(n_seqs, out_csv):
    batch_size = params['batch_size']
    n_actual = batch_size * math.ceil(n_seqs/batch_size)
    z_sample = np.random.normal(0, 1, size = (n_actual, params['latent_dim']))
    amino_acid_arr = decoder.predict(z_sample)    
    padded_array = np.array([conversion.onehot_to_seq(amino_acid_arr[i]) for i in range(amino_acid_arr.shape[0])])
    unpadded_array = [conversion.unpad(padded_array[i]) for i in range(len(padded_array))]
    pd.DataFrame({'amino_acid': unpadded_array}).to_csv(out_csv, index = False)


vae, encoder, decoder, callbacks = create_model(params)
'''
start = time.clock()
#train_file = '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/cohort2_standard_output/count_in_64/training-sequences.csv'
#train_file = '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_main/_output_emerson-2017-03-04/emerson-2017-03-04.train/training.csv'
fit(args.train_file, 0.1, str(args.beta) + '_vae_seq_bestweight.h5', str(args.beta))
print((time.clock() - start)/60)

'''
model_weights = args.model_weights
encoder.load_weights(model_weights, by_name = True)
decoder.load_weights(model_weights, by_name = True)

#generate(10000, 'vae_seq_generated.csv')
pvae(100, args.input, args.output)
'''
def add_pcs():
    """
    Add principal component information to a copy TCR data frame.
    """
    df = pd.read_csv('/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_main/_output_emerson-2017-03-04/merged.agg.csv.bz2')
    df = df.loc[(df['beta'] == 0.75) & (df['model'] == 'basic'),]
    data = conversion.unpadded_tcrbs_to_onehot(df, params['max_cdr3_len'], 'middle')
    cdr3 = np.array(data.iloc[:, 0].tolist())
    z_mean,_ = encoder.predict(cdr3)
    
    pca = PCA(n_components=2)
    pca.fit(z_mean)
    z_mean_pcs = pca.transform(z_mean)

    df = pd.DataFrame(df)
    df['pc_1'] = z_mean_pcs[:, 0]
    df['pc_2'] = z_mean_pcs[:, 1]
    return df    
    
add_pcs().to_csv('pcs2.csv', index=False)
'''
