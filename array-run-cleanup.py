# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import json
import numpy as np
import pandas as pd
import os

print("Import success")

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

# load config
with open("config.json") as file:
    config = json.load(file)

# load data
counts = np.load(f"./data/{config['data_name']}.npy")

# size
repeats = counts.shape[2]

# result dataframe
result_df = pd.DataFrame(
    columns=metrics
)

# loop
for r in range(repeats):

    # load results
    r_df = pd.read_csv(f"./data/{config['data_name']}-{config['model_name']}-{r}.csv", index_col=0)

    # store results
    result_df = pd.concat([result_df, r_df])

# save
result_df.to_csv(f"./data/{config['data_name']}-{config['model_name']}.csv")

# remove files
for r in range(repeats):

    # remove data file
    os.remove(f"./data/{config['data_name']}-{config['model_name']}-{r}.csv")
