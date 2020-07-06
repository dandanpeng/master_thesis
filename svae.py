#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:28:28 2020

@author: pengdandan
"""
import click, time
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
from sklearn.decomposition import PCA
import vampire.common as common
import vampire.xcr_vector_conversion as conversion

from vampire.custom_keras import EmbedViaMatrix

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str)
parser.add_argument('--model_weights', type = str)
parser.add_argument('--gpu', type = str)
parser.add_argument('--input', type = str)
parser.add_argument('--output', type = str)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
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
        # Model parameters.
        latent_dim=20,
        dense_nodes=70,
        aa_embedding_dim=21,
        v_gene_embedding_dim=30,
        j_gene_embedding_dim=13,
        beta=0.6,
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

def create_model(params):
    
    def sampling(args):
        """
        This function draws a sample from the multivariate normal defined by
        the latent variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
        # Reparameterization trick!
        return (z_mean + K.exp(z_log_var / 2) * epsilon)

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
    
    # V gene Classifier
    clf_input1 = Input(shape = (params['latent_dim'], ))
    clf_output1 = Dense(59, activation = 'softmax', name = 'v_gene_output')
    v_gene_output = clf_output1(z_l([z_mean, z_log_var]))
    
    # J gene Classfier
    clf_input2 = Input(shape = (params['latent_dim'], ))
    clf_output2 = Dense(13, activation = 'softmax', name = 'j_gene_output')    
    j_gene_output = clf_output2(z_l([z_mean, z_log_var]))
    
    encoder = Model(cdr3_input, [z_mean, z_log_var])
    decoder = Model(z_mean_input, decoder_cdr3_output)
    v_gene_clf = Model(clf_input1, clf_output1(clf_input1))
    j_gene_clf = Model(clf_input2, clf_output2(clf_input2))
    
    def reconstruction_loss(io_encoder, io_decoder):
        return params['max_cdr3_len'] * K.mean(objectives.categorical_crossentropy(cdr3_input, cdr3_output))    
    
    def kl_loss(io_encoder, io_decoder):
        return -0.5 * K.variable(params['beta']) * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    
    def identity(io_encoder, io_decoder):
        encoder_indice = K.argmax(io_encoder)
        decoder_indice = K.argmax(io_decoder)
        mask = K.not_equal(encoder_indice, 20)       
        encoder_indice_wo_gap = tf.boolean_mask(encoder_indice, mask)
        decoder_indice_wo_gap = tf.boolean_mask(decoder_indice, mask)
        return K.cast(K.equal(encoder_indice_wo_gap, decoder_indice_wo_gap), K.floatx())
    
    svae = Model(cdr3_input, [cdr3_output, v_gene_output, j_gene_output])
    svae.compile(optimizer="adam",
                loss={
                    'cdr3_output': vae_cdr3_loss,
                    'v_gene_output': keras.losses.categorical_crossentropy,
                    'j_gene_output': keras.losses.categorical_crossentropy
                    },
                loss_weights={
                    # Keep the cdr3_output weight to be 1. The weights are relative
                    # anyhow, and buried inside the vae_cdr3_loss is a beta weight that
                    # determines how much weight the KL loss has. If we keep this
                    # weight as 1 then we can interpret beta in a straightforward way.
                    "cdr3_output": 1,
                    "j_gene_output": 0.1305,
                    "v_gene_output": 0.8138
                    },
                metrics = [identity])
     
    return svae, encoder, decoder, v_gene_clf, j_gene_clf

def reinitialize_weights():
        session = K.get_session()
        for layer in svae.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
                
def fit(train_file: str, validation_split: float, best_weights_fname: str, tensorboard_log_dir: str):
    """
    Fit the vae with warmup and early stopping.
    """
    train_csv = pd.read_csv(train_file)
    train_data = conversion.unpadded_tcrbs_to_onehot(train_csv, params['max_cdr3_len'], 'middle')
    cdr3 = np.array(train_data.iloc[:, 0].tolist())
    v_gene = np.array(train_data.iloc[:, 1].tolist())
    j_gene = np.array(train_data.iloc[:, 2].tolist())    

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
        history = svae.fit(
            x=[cdr3],  # y=X for a VAE.
            y=[cdr3, v_gene, j_gene],
            epochs=1 + params['warmup_period'],
            batch_size=params['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=2)
        new_val_loss = history.history['val_loss'][-1]
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss
            svae.save_weights(best_weights_fname, overwrite=True)
    
    svae.load_weights(best_weights_fname)

    checkpoint = ModelCheckpoint(best_weights_fname, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(
        monitor=params['stopping_monitor'], patience=params['patience'], mode='min')
    callbacks = [checkpoint, early_stopping]
    if tensorboard_log_dir:
        callbacks += [keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)]
    svae_history = svae.fit(
        x=cdr3,  # y=X tfor a VAE.
        y=[cdr3, v_gene, j_gene],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2)
    return svae_history.history

def logprob_of_obs_vect(probs, obs):
    """
    Calculate the log of probability of the observations.

    :param probs: a matrix with each row giving the probability of
        observations.
    :param obs: a matrix with each row one-hot-encoding an observation.

    Kristian implemented this as
        np.sum(np.log(np.matmul(probs, obs.T).diagonal()))
    which is equivalent but harder to follow.
    """
    # Here axis=1 means sum across columns (the sum will be empty except for
    # the single nonzero entry).
    return np.sum(np.log(np.sum(probs * obs, axis=1)))

def log_pvae_importance_sample(x_df, out_ps):
    cdr3 = np.array(x_df.iloc[:, 0].tolist())
    
    z_mean, z_log_var = encoder.predict(cdr3)
    z_sd = np.sqrt(np.exp(z_log_var))
    z_sample = stats.norm.rvs(z_mean, z_sd)    
    
    aa_probs = decoder.predict(z_sample)
    v_gene_probs = v_gene_clf.predict(z_sample)
    j_gene_probs = j_gene_clf.predict(z_sample)
    aa_obs, v_gene_obs, j_gene_obs = common.cols_of_df(x_df) 
    
    for i in range(len(x_df)):
        log_p_x_given_z = \
                logprob_of_obs_vect(aa_probs[i], aa_obs[i]) + \
                np.log(np.sum(v_gene_probs[i] * v_gene_obs[i])) + \
                np.log(np.sum(j_gene_probs[i] * j_gene_obs[i]))
        log_p_z = np.sum(stats.norm.logpdf(z_sample[i], 0, 1))
        log_q_z_given_x = np.sum(stats.norm.logpdf(z_sample[i], z_mean[i], z_sd[i]))
        log_imp_weight = log_p_z - log_q_z_given_x
        out_ps[i] = log_p_x_given_z + log_imp_weight   
        
def pvae(evaluation_file, out_csv, nsamples = 500):    
    evaluation_csv = pd.read_csv(evaluation_file)
    evaluation_data = conversion.unpadded_tcrbs_to_onehot(evaluation_csv, params['max_cdr3_len'], 'middle')
    #cdr3 = np.array(evaluation_data.iloc[:, 0].tolist())
    
    log_p_x = np.zeros((nsamples, len(evaluation_data)))
    
    with click.progressbar(range(nsamples)) as bar:
        for i in bar:
            log_pvae_importance_sample(evaluation_data, log_p_x[i])
    
    avg = special.logsumexp(log_p_x, axis = 0) - np.log(nsamples)
    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index=False)
    

def generate(n_seqs, out_csv):
    z_sample = np.random.normal(0, 1, size = (n_seqs, params['latent_dim']))
    amino_acid_arr = decoder.predict(z_sample)
    v_gene_arr = v_gene_clf.predict(z_sample)
    j_gene_arr = j_gene_clf.predict(z_sample)
    
    df = conversion.onehot_to_tcrbs(amino_acid_arr, v_gene_arr, j_gene_arr)
    df.to_csv(out_csv, index=False)
    
    
svae, encoder, decoder, v_gene_clf, j_gene_clf = create_model(params)
'''
#train_file = '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/cohort2_standard_output/count/training-sequences.csv'
train_file = '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_main/_output_emerson-2017-03-04/emerson-2017-03-04.train/training.csv'
start = time.clock()
fit(train_file, 0.1, 'main_best_weights.h5', str(params['beta']))
print((time.clock() - start)/60)

'''
model_weights = args.model_weights
#model_weights = str(params['beta']) + '_best_weights.h5'

encoder.load_weights(model_weights, by_name = True)
decoder.load_weights(model_weights, by_name = True)
v_gene_clf.load_weights(model_weights, by_name = True)
j_gene_clf.load_weights(model_weights, by_name = True)

#decoder.save('svae_decoder.h5')
#generate(10000, str(params['beta']) + '_generated.csv')

pvae(args.input, args.output, 100)

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
    
add_pcs().to_csv('pcs.csv', index=False)

test_file = '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/cohort2_standard_output/count/basic/evaluation-sequences-from-test.csv'
pvae(test_file, '0.6_test.pvae.csv', 100)


svae.save_weights('bestweight.h5', overwrite=True)

pvae(args.evaluation_file, args.out_csv)
  
'''  
    
    
    
    
    
    
