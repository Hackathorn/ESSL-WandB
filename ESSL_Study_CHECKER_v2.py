# %% [markdown]
# # ESSL study of CHECKER dataset using WandB monitoring
# 
# <a href="https://colab.research.google.com/github/Hackathorn/ESSL-WandB/blob/master/ESSL_Study_CHECKER_v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Setup ESSL repro in Colab

# %%
# ############# Setup ESSL in Colab
# %%capture

# !git clone https://github.com/Hackathorn/ESSL-WandB/
# %cd ESSL-WandB

# !pip install umap-learn
# !pip install wandb --upgrade

# %% [markdown]
# ## Import & Setup modules

# %%
from pathlib import Path
import os, pprint
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

# import ESSL Utils modules
from utils.sample import    create_samples_from_CHECKER

from utils.point import     create_points, \
                            add_posLowD_to_points, \
                            plot_pos2D_points

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# remove slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

import wandb
# wandb.login()     Not Needed??? 


# %%

sweep_config = {
    'name': 'NONE',         # set in Main routine below
    'method':   'random',
    'metric':   {
        'name': 'recon_loss',
        'goal': 'minimize',        
    },
    'parameters':   {
        # params for Samples
        'dataset':  { 
            'values': [ "CHECKER" ] 
            }, 
        'n_samples':  { 
            'value': 3000
            },
        'img_size':  { 
            'values': [ 32 ] 
            },
        'classes':  { 
            'values': [ 10 ] 
            },
        'blk_size':  { 
            'values': [ 2 ] 
            },
        'noise':  { 
            'values': [ 0.0 ] 
            },
        # params for Points
        'model_arch':  { 
            'values': [ 'VAE1' ] 
            },
        'epochs':  { 
            'values': [ 10 ] 
            },
        'batch_size':  { 
            'values': [ 256 ] 
            },
        'learning_rate':  { 
            'values': [ 0.005 ] 
            },
        'lat_dim':  { 
            'values': [ 16 ] 
            },
        'beta':  { 
            'values': [ 1.0 ] 
            },
        
    }
}
print('---WandB Sweep Configuration---')
pprint.pprint(sweep_config)

## Define Hyperparameters

# hyperparameters = dict( 
#         # params for Samples
#         dataset="CHECKER", 
#         n_samples=3000,
#         img_size=32,
#         classes=10,
#         blk_size=2,
#         noise=None,
#         # params for Points
#         model_arch="VAE1",
#         epochs=5,
#         batch_size=256,
#         learning_rate=0.005,
#         lat_dim=16,
#         beta=1,
#     )

# %% [markdown]
# ## Define ESSL Pileline

# %%
from utils.sample import create_samples
from utils.point import create_points       #, analyze_points

def ESSL_pipeline():

    # tell wandb to get started             NOTE: use mode="disabled" for debugging
    with wandb.init():
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config
      
      # create samples from dataset
      samples_df = create_samples(config)
      print(samples_df.info())
      
      # create points from samples
      points_df = create_points(config, samples_df)
      print(points_df.info())

# %% [markdown]
# ### Create Samples

# %%
from utils.sample import create_samples_from_CHECKER #, analyze_sample

def create_samples(config):
    
    samples_df = create_samples_from_CHECKER(config.n_samples, 
                                             config.img_size, 
                                             n_classes=config.classes, 
                                             n_blk_size=config.blk_size,
                                             noise=config.noise,
    )
    # wandb.log({'sample_df': sample_df})
   
    return samples_df

# %% [markdown]
# ### Create Points

# %%
from utils.point import VAE_Train_Dataset, VAE, train_VAE
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def create_points(config, samples_df):
    
    # instantiate custom DataSet for CHECKER and DataLoader
    train_dataset = VAE_Train_Dataset(samples_df, transform=None)
    
    # instantiate training loader 
    batch_size = config.n_samples // 10
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config.batch_size, 
                            shuffle=True, num_workers=2)
    print('Shape of Train_Loader Sample =', next(iter(train_loader))[0].shape)

    # instantiate model and print
    n_channels = 1
    model = VAE(n_channels, config.img_size, config.lat_dim).to(device)
    print(model)
    
    # define optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()    # >>>> TODO versus F.mse_loss

    # train VAE model for optimal D-dim latent space
    train_VAE(config,
              model, criterion, optimizer, 
              train_loader,
              device,
              )
    
    # and test its final performance
    # test(model, test_loader)

    points_df = pd.DataFrame()
    return points_df

# %%
############### Execute ESSL Pipeline with Sweep Configuration

if __name__ == '__main__':
    
    project = Path(__file__).stem
    sweep_config['name'] = "Initial Single-Run Sweep #2"
    
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, function=ESSL_pipeline, count=1)
