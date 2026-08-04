# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

#import argparse
#import json
import scvi
import numpy as np
import pandas as pd
import scipy
import anndata as ad
import time

print("Import success")

# ------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------

# script arguments
#parser = argparse.ArgumentParser()
#parser.add_argument("--repeat", type=int)

# parse
#args = parser.parse_args()

# load config
#with open("config.json") as file:
#    config = json.load(file)

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def model_statistics_bootstrap(model, samples=100, confidence=0.95):

    # settings
    alpha = 1 - confidence

    # sample
    counts = model.posterior_predictive_sample(n_samples=samples, gene_list=["0", "1"]).todense()

    # compute mean statistics
    means = np.mean(counts, axis=0)
    mean_interval = np.quantile(means, [(alpha / 2), 1 - (alpha / 2)], axis=1)
    mean_mean = np.mean(means, axis=1)
    
    # compute variance statistics
    varx = np.var(counts, axis=0)
    varx_interval = np.quantile(varx, [(alpha / 2), 1 - (alpha / 2)], axis=1)
    varx_mean = np.mean(varx, axis=1)

    # compute correlation statistics
    prsn = np.empty((samples))
    sprm = np.empty((samples))
    for s in range(samples):
        prsn[s] = scipy.stats.pearsonr(counts[:, 0, s], counts[:, 1, s]).statistic
        sprm[s] = scipy.stats.spearmanr(counts[:, 0, s], counts[:, 1, s]).statistic
    prsn_interval = np.quantile(prsn, [(alpha / 2), 1 - (alpha / 2)], axis=0)
    sprm_interval = np.quantile(sprm, [(alpha / 2), 1 - (alpha / 2)], axis=0)
    prsn_mean = np.mean(prsn)
    sprm_mean = np.mean(sprm)

    # collect data
    data_dict = {
        'mean_interval': mean_interval,
        'mean_mean': mean_mean,
        'vars_interval': varx_interval,
        'vars_mean': varx_mean,
        'prsn_interval': prsn_interval,
        'prsn_mean': prsn_mean,
        'sprm_interval': sprm_interval,
        'sprm_mean': sprm_mean
    }

    return data_dict

def train_scVI_model(counts, model_kwargs={}, train_kwargs={}):
    '''
    Train scVI model on a sample from data distribution with given params
    and record metrics

    counts: training data
    model_kwargs: keyword arguments for model
    train_kwargs: keyword arguments for training
    '''

    # by default compute validation loss
    if not ('check_val_every_n_epoch' in train_kwargs.keys()):
        train_kwargs['check_val_every_n_epoch'] = 1

    # setup anndata
    counts_sparse = scipy.sparse.csr_matrix(counts)
    adata = ad.AnnData(counts)
    adata.layers["counts"] = counts_sparse
    scvi.model.SCVI.setup_anndata(adata, layer="counts")

    # create model
    model = scvi.model.SCVI(adata, **model_kwargs)

    # train model
    s = time.time()
    model.train(**train_kwargs)
    t = time.time() - s

    # compute model statistics
    stats = model_statistics_bootstrap(model)

    # collect results
    result_dict = {
        'training_time': t,
        'elbo_train': float(model.history['elbo_train'].iloc[-1, 0]),
        'reconstruction_loss_train': float(model.history['reconstruction_loss_train'].iloc[-1, 0]),
        'kl_local_train': float(model.history['kl_local_train'].iloc[-1, 0]),
        'elbo_validation': float(model.history['elbo_validation'].iloc[-1, 0]),
        'reconstruction_loss_validation': float(model.history['reconstruction_loss_validation'].iloc[-1, 0]),
        'kl_local_validation': float(model.history['kl_local_validation'].iloc[-1, 0]),
        'mean_1_lb': float(stats['mean_interval'][0, 0]),
        'mean_1_ub': float(stats['mean_interval'][1, 0]),
        'mean_2_lb': float(stats['mean_interval'][0, 1]),
        'mean_2_ub': float(stats['mean_interval'][1, 1]),
        'mean_1_point': float(stats['mean_mean'][0]),
        'mean_2_point': float(stats['mean_mean'][1]),
        'vars_1_lb': float(stats['vars_interval'][0, 0]),
        'vars_1_ub': float(stats['vars_interval'][1, 0]),
        'vars_2_lb': float(stats['vars_interval'][0, 1]),
        'vars_2_ub': float(stats['vars_interval'][1, 1]),
        'vars_1_point': float(stats['vars_mean'][0]),
        'vars_2_point': float(stats['vars_mean'][1]),
        'prsn_lb': float(stats['prsn_interval'][0]),
        'prsn_ub': float(stats['prsn_interval'][1]),
        'prsn_point': float(stats['prsn_mean']),
        'sprm_lb': float(stats['sprm_interval'][0]),
        'sprm_ub': float(stats['sprm_interval'][1]),
        'sprm_point': float(stats['sprm_mean'])
    }
    
    return result_dict

# ------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------

# define model metrics
metrics = [
    'training_time',
    'elbo_train',
    'reconstruction_loss_train',
    'kl_local_train',
    'elbo_validation',
    'reconstruction_loss_validation',
    'kl_local_validation',
    'mean_1_lb',
    'mean_1_ub',
    'mean_2_lb',
    'mean_2_ub',
    'mean_1_point',
    'mean_2_point',
    'vars_1_lb',
    'vars_1_ub',
    'vars_2_lb',
    'vars_2_ub',
    'vars_1_point',
    'vars_2_point',
    'prsn_lb',
    'prsn_ub',
    'prsn_point',
    'sprm_lb',
    'sprm_ub',
    'sprm_point'
]

# model settings
model_kwargs = {
    'gene_likelihood': 'poisson', # config['gene_likelihood'],
    'n_hidden': 16, # config['n_hidden'],
    'n_latent': 2, # config['n_latent'],
    'n_layers': 1, # config['n_layers'],
    'use_observed_lib_size': True # config['use_observed_lib_size']
}

# training settings
train_kwargs={
    'max_epochs': 100, # config['max_epochs'],
    'early_stopping': False # config['early_stopping']
}

# load data
counts = np.load(f"./data/indep-poi.npy") # np.load(f"./data/{config['data_name']}.npy")

print("Data loading success")

# result dataframe
result_df = pd.DataFrame(
    columns=metrics
)

# run
# result = train_scVI_model(counts[:, :, args.repeat], model_kwargs=model_kwargs, train_kwargs=train_kwargs)
result = train_scVI_model(counts[:, :, 0], model_kwargs=model_kwargs, train_kwargs=train_kwargs)

print("Training success")

# store results
# result_df = pd.DataFrame(result, index=[args.repeat])
result_df = pd.DataFrame(result, index=[0])

# save
# result_df.to_csv(f"./data/{config['data_name']}-{config['model_name']}-{args.repeat}.csv")
result_df.to_csv(f"./data/indep-poi-test-0.csv")
