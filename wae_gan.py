"""
Kristian's original 2-layer VAE.

Model diagram with 35 latent dimensions and 100 dense nodes:
https://user-images.githubusercontent.com/112708/48358766-4f7a7e00-e650-11e8-9bab-d7a294548100.png
"""
import numpy as np
import pandas as pd

import keras
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Lambda, Input, Reshape
from keras import backend as K
from keras import objectives

import tensorflow as tf

from vampire.custom_keras import EmbedViaMatrix
import vampire.common as common
import vampire.xcr_vector_conversion as conversion

import scipy.special as special
import scipy.stats as stats

from datetime import datetime

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        model='wae_gan',
        # Model parameters.
        latent_dim=20,
        dense_nodes=75,
        aa_embedding_dim=21,
        v_gene_embedding_dim=30,
        j_gene_embedding_dim=13,
        beta=0.75,
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
        epochs=200,
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


def vae_cdr3_loss(io_encoder, io_decoder):
    """
    The loss function is the sum of the cross-entropy and KL divergence. KL
    gets a weight of beta.
    """
    # Here we multiply by the number of sites, so that we have a
    # total loss across the sites rather than a mean loss.
    xent_loss = params['max_cdr3_len']* K.mean(objectives.categorical_crossentropy(io_encoder, io_decoder))
    #kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #kl_loss *= K.variable(params['beta'])
    return xent_loss

def identity(io_encoder, io_decoder):
    """
    The identity is the percentage of same amino acid at each position excluded '-' (gap).
    """
    encoder_indice = K.argmax(io_encoder)
    decoder_indice = K.argmax(io_decoder)
    mask = K.not_equal(encoder_indice, 20)  
    encoder_indice_wo_gap = tf.boolean_mask(encoder_indice, mask)
    decoder_indice_wo_gap = tf.boolean_mask(decoder_indice, mask)
    
    return K.cast(K.equal(encoder_indice_wo_gap, decoder_indice_wo_gap), K.floatx())  

def create_model(params):
    # Input:
    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])
    cdr3_input = Input(shape=cdr3_input_shape, name='cdr3_input')
    v_gene_input = Input(shape=(params['n_v_genes'], ), name='v_gene_input')
    j_gene_input = Input(shape=(params['n_j_genes'], ), name='j_gene_input')
    
    # Encoding layers:
    cdr3_embedding = EmbedViaMatrix(params['aa_embedding_dim'], name='cdr3_embedding')(cdr3_input)
    cdr3_embedding_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='cdr3_embedding_flat')(cdr3_embedding)
    v_gene_embedding = Dense(params['v_gene_embedding_dim'], name='v_gene_embedding')(v_gene_input)
    j_gene_embedding = Dense(params['j_gene_embedding_dim'], name='j_gene_embedding')(j_gene_input)
    merged_embedding = keras.layers.concatenate([cdr3_embedding_flat, v_gene_embedding, j_gene_embedding],
                                                name='merged_embedding')
    encoder_dense_1 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_1')(merged_embedding)
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
    v_gene_output_l = Dense(params['n_v_genes'], activation='softmax', name='v_gene_output')
    j_gene_output_l = Dense(params['n_j_genes'], activation='softmax', name='j_gene_output')
    
    post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var])))
    cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(post_decoder)))
    v_gene_output = v_gene_output_l(post_decoder)
    j_gene_output = j_gene_output_l(post_decoder)
    
    # Define the decoder components separately so we can have it as its own model.
    z_mean_input = Input(shape=(params['latent_dim'], ))
    decoder_post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_mean_input))
    decoder_cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(decoder_post_decoder)))
    decoder_v_gene_output = v_gene_output_l(decoder_post_decoder)
    decoder_j_gene_output = j_gene_output_l(decoder_post_decoder)
    
    # Discriminator layers:
    disc_dense_1_l = Dense(20, activation='relu')
    disc_dense_2_l = Dense(20, activation='relu')
    disc_output_l = Dense(1, activation='sigmoid', name = 'disc_output')
    disc_output = disc_output_l(disc_dense_2_l(disc_dense_1_l(z_l([z_mean, z_log_var]))))
    
    # Define the discriminator separately so we can have it as its own model.
    d_input = Input(shape=(params['latent_dim'], ))
    d_output = disc_output_l(disc_dense_2_l(disc_dense_1_l(d_input)))
    
    encoder = Model([cdr3_input, v_gene_input, j_gene_input], [z_mean, z_log_var])
    decoder = Model(z_mean_input, [decoder_cdr3_output, decoder_v_gene_output, decoder_j_gene_output])
    
    vae = Model([cdr3_input, v_gene_input, j_gene_input], [cdr3_output, v_gene_output, j_gene_output])
    vae.compile(
         optimizer="adam",
         loss={
             'cdr3_output': vae_cdr3_loss,
             'v_gene_output': keras.losses.categorical_crossentropy,
             'j_gene_output': keras.losses.categorical_crossentropy,
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
         metrics = [identity]
         )
    
    discriminator = Model(d_input, d_output)
    discriminator.compile(
        optimizer = "adam",
        loss = 'binary_crossentropy'
        ) 
    
    discriminator.trainable = False
    generator = Model([cdr3_input, v_gene_input, j_gene_input], 
                      discriminator((z_l(encoder([cdr3_input, v_gene_input, j_gene_input])))))
    
    generator.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy')
    
    return vae, discriminator, generator, encoder, decoder
    
    
def train(train_file, params):
    train_csv = pd.read_csv(train_file)
    train_data = conversion.unpadded_tcrbs_to_onehot(train_csv, params['max_cdr3_len'], 'middle')
    cdr3 = np.array(train_data.iloc[:1000, 0].tolist())
    v_gene = np.array(train_data.iloc[:1000, 1].tolist())
    j_gene = np.array(train_data.iloc[:1000, 2].tolist())    
    
    vae, discriminator, generator, encoder, decoder = create_model(params)
    
    past = datetime.now()
    for epoch in np.arange(1, params['epochs'] +1):
        vae_loss = []
        discriminator_loss = []
        generator_loss = []
        pred_identity = []
        for batch in np.arange(len(train_data) / params['batch_size']):
            start = int(batch * params['batch_size'])
            end = int(start + params['batch_size'])
            cdr3_samples = cdr3[start:end]
            v_gene_samples = v_gene[start:end]
            j_gene_samples = j_gene[start:end]
            samples = [cdr3_samples, v_gene_samples, j_gene_samples]
            vae_history = vae.fit(samples, samples, 
                                  epochs = 1,
                                  batch_size = params['batch_size'], 
                                  validation_split = 0.0,
                                  verbose = 0)
            vae_loss.append(vae_history.history['loss'])
            pred_identity.append(vae_history.history['cdr3_output_identity'])
            
            # Train Discriminator
            fake_latent = K.eval(sampling(encoder.predict(samples)))
            real_latent = np.random.normal(size = (params['batch_size'], params['latent_dim'])) 
            
            d_real_history = discriminator.fit(real_latent, np.ones((params['batch_size'], 1)), 
                                            epochs = 1, batch_size = params['batch_size'], 
                                            validation_split = 0.0, verbose = 0)
            d_fake_history = discriminator.fit(fake_latent, np.zeros((params['batch_size'], 1)),
                                               epochs = 1, batch_size = params['batch_size'],
                                               validation_split = 0.0, verbose = 0)
            discriminator_loss.append(0.5 * np.add(d_real_history.history['loss'], d_fake_history.history['loss']))
            
 
            # Train Generator
            generator_history = generator.fit(samples, np.ones((params['batch_size'], 1)),
                                   epochs = 1, 
                                   batch_size = params['batch_size'], 
                                   validation_split = 0.0,
                                   verbose = 0)
            
            generator_loss.append(generator_history.history['loss'])     
        
        now = datetime.now()
        print("\nEpoch {}/{} - {:.1f}s".format(epoch, params['epochs'], (now - past).total_seconds()))
        print("VAE Loss: {}".format(np.mean(vae_loss)))
        print("Discriminator Loss: {}".format(np.mean(discriminator_loss)))
        print("Generator Loss: {}".format(np.mean(generator_loss)))
        print("Identity: {}".format(np.mean(pred_identity)))
    
        #if epoch % 10 == 0:
            #print("\nSaving models...")
            #encoder.save('encoder.h5')
            #decoder.save('decoder.h5')
            #discriminator.save('discriminator.h5')

def logprob_of_obs_vect(probs, obs):
    return np.sum(np.log(np.sum(probs * obs, axis=1)))

def pvae(nsamples, model_weights, test_csv, out_csv):
    vae.load_model(model_weights)
    
    df = pd.read_csv(test_csv)
    df_x = conversion.unpadded_tcrbs_to_onehot(df, params['max_cdr3_len'])
    cdr3 = np.array(df_x.iloc[:, 0].tolist())
    v_gene = np.array(df_x.iloc[:, 1].tolist())
    j_gene = np.array(df_x.iloc[:, 2].tolist())
 
    log_p_x = np.zeros((nsamples, len(df_x)))
    
    for j in range(len(log_p_x)):
        assert (len(df_x) == len(log_p_x[j]))
       
        z_mean, z_log_var = encoder.predict([cdr3, v_gene, j_gene])
        z_sd = np.sqrt(np.exp(z_log_var))
        z_sample = stats.norm.rvs(z_mean, z_sd)
        aa_probs, v_gene_probs, j_gene_probs = decoder.predict(z_sample)

        aa_obs, v_gene_obs, j_gene_obs = common.cols_of_df(df_x)
        
        for i in range(len(df_x)):
              log_p_x_given_z = \
                  logprob_of_obs_vect(aa_probs[i], aa_obs[i]) + \
                  np.log(np.sum(v_gene_probs[i] * v_gene_obs[i])) + \
                  np.log(np.sum(j_gene_probs[i] * j_gene_obs[i]))

              log_p_z = np.sum(stats.norm.logpdf(z_sample[i], 0, 1))
              log_q_z_given_x = np.sum(stats.norm.logpdf(z_sample[i], z_mean[i], z_sd[i]))
              log_imp_weight = log_p_z - log_q_z_given_x
              log_p_x[j][i] = log_p_x_given_z + log_imp_weight
        print(j/len(log_p_x))
    avg = special.logsumexp(log_p_x, axis = 0) - np.log(nsamples)
    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index = False)


def generate(n_seqs, decoder_weights, out_csv):
    decoder = load_model(decoder_weights)
    z_sample = np.random.normal(0, 1, size = (n_seqs, params['latent_dim']))
    amino_acid_arr, v_gene_arr, j_gene_arr = decoder.predict(z_sample)
    sequences = conversion.onehot_to_tcrbs(amino_acid_arr, v_gene_arr, j_gene_arr)
    return sequences.to_csv(out_csv, index = False)

train('/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_main/_output_emerson-2017-03-04/emerson-2017-03-04.train/training.csv',
      default_params())


'''
pvae(500, '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/wae.weights', 
     '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/evaluation-sequences-from-train.csv', 
     '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/evaluation-sequences-from-train.pvae.csv')

pvae(500, '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/wae.weights', 
     '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/evaluation-sequences-from-test.csv', 
     '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_freq/wae_mini_output/count_in_64/wae/evaluation-sequences-from-test.pvae.csv')

generate(10000, 'decoder.h5', '/mnt/ds3lab-scratch/pengd/thesis/tools/vampire/vampire/pipe_main/wae_mini_output/wae-generated.csv')
'''
