# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

data_list = [
    "real-data"
]

model_list = [
    {"model_name": "NB-16-2-1-400-soft", "gene_likelihood": "nb", "n_hidden": 16, "n_latent": 2, "n_layers": 1, "lib": "soft"}
]

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import argparse
import json
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
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int)

# parse
args = parser.parse_args()

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

def train_scVI_model(counts, model_kwargs={}, train_kwargs={}, lib="obs"):
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

    # (default) observed library size
    if lib == "obs":
        adata = ad.AnnData(counts)
        scvi.model.SCVI.setup_anndata(adata)

    # observed library size: pad with constant gene
    elif lib == "pad":
        adata = ad.AnnData(np.hstack([counts, np.ones(counts.shape[0]).reshape(-1, 1)]))
        scvi.model.SCVI.setup_anndata(adata)

    # fixed library size: constant of mean total cell counts
    elif lib == "const":
        adata = ad.AnnData(counts)
        adata.obs['size_factor_constant'] = np.ones(counts.shape[0]) * np.mean(counts) * 2
        scvi.model.SCVI.setup_anndata(adata, size_factor_key="size_factor_constant")

    # (real data only) fixed library size: total counts over all genes (rescaled to match mean total cell counts of pair)
    elif (lib == "real" and data_name == "real-data"):

        # load real total counts: scale to have same mean as gene pair total counts
        total_counts = np.load("./data/total-counts.npy")
        scaling = 2 * np.mean(counts) / np.mean(total_counts)

        adata = ad.AnnData(counts)
        adata.obs['size_factor_real'] = total_counts * scaling
        scvi.model.SCVI.setup_anndata(adata, size_factor_key="size_factor_real")

    # fix to observed library size: l_n value unchanged, but changes softmax to softplus
    elif lib == "soft":
        adata = ad.AnnData(counts)
        adata.obs['size_factor_obs'] = np.mean(counts, axis=1)
        scvi.model.SCVI.setup_anndata(adata, size_factor_key="size_factor_obs")

    else:
        raise Exception("Invalid library size method")

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

# ------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------

# convert index to data, model, repeat (subtract 1 to start at 0)
idx = args.index - 1

num_datas = len(data_list)
num_models = len(model_list)
num_repeats = 30

data_idx = (idx // (num_repeats * num_models)) % num_datas
model_idx = (idx // (num_repeats)) % num_models
repeat_idx = (idx) % num_repeats

data_name = data_list[data_idx]
model_dict = model_list[model_idx]

# model settings
model_kwargs = {
    'gene_likelihood': model_dict['gene_likelihood'],
    'n_hidden': model_dict['n_hidden'],
    'n_latent': model_dict['n_latent'],
    'n_layers': model_dict['n_layers'],
    'use_observed_lib_size': True
}

# training settings
train_kwargs={
    'max_epochs': 400,
    'early_stopping': False
}

# load data
counts = np.load(f"./data/{data_name}.npy")

print("Data loading success")

# result dataframe
result_df = pd.DataFrame(
    columns=metrics
)

# run
result = train_scVI_model(counts[:, :, repeat_idx], model_kwargs=model_kwargs, train_kwargs=train_kwargs, lib=model_dict['lib'])

print("Training success")

# store results
result_df = pd.DataFrame(result, index=[repeat_idx])

# save
result_df.to_csv(f"./data/{data_name}-{model_dict['model_name']}-{repeat_idx}.csv")
