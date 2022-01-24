# %% [markdown]
'''
# STEP2 - Embed Samples into Latent Space as Points  

- Objective: Create LS with good similiarity dispersion, both globally & locally

- Hyperparams: hparams = {'epochs': [40], 'image_size': [64,128,256], 'latent_dim': [64,128,256]}

- Metrics: Reconstruction error (MSE loss comparing before/after images), 

Derived from... RASBT-STAT453 Spring2021 L17 4_VAE_celeba-inspect-latent.ipynb
https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/2_VAE_celeba-sigmoid_mse.ipynb
'''

# %%
# Import packages
import enum
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import time, os, random

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn.functional as F
import umap

from utils.experiment import save_dataframe
# from RASBT_helper_utils import set_deterministic, set_all_seeds
# from RASBT_helper_plotting import plot_training_loss
# from RASBT_helper_plotting import plot_generated_images
# from RASBT_helper_plotting import plot_latent_space_with_labels

# %%
# from RASBT-STAT453 Spring2021 L17 helper_utils

def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
    
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
# set training non-random-ness
RANDOM_SEED = 42
set_deterministic           # from RASBT_helper_utils
set_all_seeds(RANDOM_SEED)  # from RASBT_helper_utils

# setup CPU/GPU Device          # >>>> TODO make DEVICE a hyperparam
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')

# Local Hyperparameters
LEARNING_RATE = 0.0005
# NUM_EPOCHS = 400
NUM_CHANNELS = 1
IMAGE_SIZE = 128
# NUM_LATENT_DIMS = 200

# %%
# define VAE training dataset from samples_df

class VAE_Train_Dataset(Dataset):
    '''
    convert file paths to PNG images to samples for custom Dataset
    '''

    # def __init__(self, samples_df, preprocess=True, transform=None):
    def __init__(self, samples_df, transform=None):
        """__init__ Initialize instance of train_dataset

        Args:
            samples_df (df): sample info either raw (path to PNG) or as preprocess (np.array)
            preprocess (str, optional): raw (false) or preprocess in sample.py (true). Defaults to None.
            transform (torch transform, optional): To tensor & normalize. Defaults to None.
        """
        self.samples_df = samples_df
        # self.preprocess = preprocess  >>> TODO remove!!! samples_df responsible
        # self.samples_df = samples_df
        self.transform = transform
        # get len of samples_df
        self.data_len = len(samples_df.index)

        # if not preprocess:    >>> TODO remove!!! samples_df responsible
        #     self.file_list = (samples_df['dataset_name'] + '\\'+samples_df['status_folder'] + \
        #         '\\' + samples_df['class_folder'] + '\\'+samples_df['img_name']).tolist()

        self.samples = np.stack(samples_df['img_array'])

        # set label_list to map to scalar
        # labels = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']   # OLD
        # self.label_list = [labels.index(label) for label in samples_df['class_folder']]
        self.label_list = [label for label in samples_df['img_label']]

    def __len__(self):

        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        # if self.preprocess:       >>> TODO remove!!! samples_df responsible
        #     train_data = self.samples[idx, :].squeeze()
        # else:
        #     img = Image.open(self.file_list[idx])
        #     train_data = np.array(ImageOps.invert(img))
        train_data = self.samples[idx, :].squeeze()

        label_data = self.label_list[idx]

        if self.transform is None:
            self.transform = transforms.ToTensor()
        train_data = self.transform(train_data)
        
        return train_data, label_data

# %%
# MODEL derived from RASBT-STAT453 Spring2021 L17 4_VAE_celeba-inspect-latent.ipynb
#   assumes image size 128x128, extended to image sizes 32, 64, 96, 128

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, img_height, img_width, *args):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width

    def forward(self, x):
        return x[:, :, :self.img_height, :self.img_width]

# dict table for shape of encoder output that depends on image size
# was designed for 128x128
# assumes square images with size = 32, 64, 96, 128 but not >128! Why? >>>> TODO flex convnet params
CONV_SHAPE_TABLE = {32*i: 256*i**2 for i in range(1,5)}

class VAE(nn.Module):
    def __init__(self, num_channels, img_size, num_latent_dims):
        super().__init__()

        self.encoder_out_shape = 0
        if img_size in CONV_SHAPE_TABLE:
            self.encoder_out_shape = CONV_SHAPE_TABLE[img_size]
        else:
            print(f'    ERROR: img_size not multiple of 32x32 <= 128x128 square')

        self.no_channels = self.encoder_out_shape // 64 # get no_channels for decoder input
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Flatten(),
        )    
        
        self.z_mean = torch.nn.Linear(self.encoder_out_shape, num_latent_dims)
        self.z_log_var = torch.nn.Linear(self.encoder_out_shape, num_latent_dims)
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(num_latent_dims, self.encoder_out_shape),
            # Reshape(-1, 64, 8, 8),    # for img_size = 128 and encode_out = 4096
            Reshape(-1, self.no_channels, 8, 8),
            #
            nn.ConvTranspose2d(self.no_channels, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(32, num_channels, stride=2, kernel_size=3, padding=1),
            #
            Trim(img_size, img_size),  # 3x129x129 -> 3x128x128  >>>> TODO Needed????  YES! WHY?
            nn.Sigmoid()
        )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

# %%
################# compute epoch loss for VAE

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)[3]     # extract model forward.decoded from 4-tuple
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss 
 
# %% -------------------------------------------------------------------------------
################# train VAE model
# derived from... RASBT-STAT453 Spring2021 L17 4_VAE_celeba-inspect-latent.ipynb
import wandb
def train_VAE(config, 
              model, criterion, optimizer, 
              train_loader, 
              device, 
            ):

    logging_interval = 1
    skip_epoch_stats = False
    example_ct = 0          # number of samples seen
    
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    loss_fn = F.mse_loss    # set loss function >>>> TODO reconcil with W&B 'criterion'

    training_start = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            example_ct += len(features)
            features = features.to(device)

            # forward proprogation
            encoded, z_mean, z_log_var, decoded = model(features)

            if features.shape != decoded.shape:
                print('>>> ERROR: features+decoded shapes NOT EQUAL ', features.shape, decoded.shape)
            
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            # TODO https://medium.com/@chengjing/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023# 
            loss = pixelwise + (config.beta * kl_div)
            
            # back-proprogation
            optimizer.zero_grad()   # zero previous gradients 
            loss.backward()         # calculate new ones
            optimizer.step()        # step backward updating weights & biases

            # LOGGING BATCH
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                epoch_duration = time.time() - epoch_start
                print('    Epoch: %03d/%03d | Batch %04d/%04d | Loss: %4.4f | Duration: %.3f sec'
                      % (epoch+1, config.epochs, batch_idx, len(train_loader), loss, epoch_duration)) 

    # LOGGING EPOCH
        model.eval()
        
        with torch.set_grad_enabled(False):  # save memory during inference
            
            train_loss = compute_epoch_loss_autoencoder(
                            model, train_loader, loss_fn, device)
            log_dict['train_combined_loss_per_epoch'].append(train_loss.item())
            wandb.log({'recon_loss': train_loss.item()})

        epoch_duration = time.time() - epoch_start
        wandb.log({"epoch": config.epochs, "loss": train_loss}, step=example_ct)
        print('Epoch: %03d/%03d | Loss: %4.3f | Duration: %.3f sec' % 
                (epoch+1, config.epochs, train_loss, epoch_duration))
            
    print('Training Time: %.1f min' % ((time.time() - training_start)/60))
    # if save_model is not None:
    #     torch.save(model.state_dict(), save_model)
    
    return log_dict
# %%
#################################################################
# plot learning curve
def plot_learning_curve(run_folder, log_dict, epochs, num_samples):

    t_loss = np.array(log_dict['train_combined_loss_per_batch']) / num_samples
    k_loss = np.array(log_dict['train_kl_loss_per_batch']) / num_samples
    v_loss = np.array(log_dict['train_reconstruction_loss_per_batch']) / num_samples

    iter_per_epoch = len(t_loss) // epochs

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparam from run_folder

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle('Point Learning Curve - ' + title_str, fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    ax.set_title(f'Loss mean={t_loss.mean():1.4f}  min={t_loss.min():1.4f}  max={t_loss.max():1.4}')

    plt.suptitle(f'Learning Curve -- {title_str}', fontsize=12, fontweight='bold')
    plt.xlabel(f'Batch Iterations over {num_samples} Samples with {epochs} Epochs')
    plt.ylabel("MSE Loss")
    plt.plot(t_loss, label="Combined with KL Loss", marker='.', linestyle = 'None')
    plt.plot(v_loss, label="Reconstruction Loss", marker='.', linestyle = 'None')
    plt.legend(loc='upper right', shadow=True)
    # plt.ylim([0, 0.7 * max(t_loss)])         # TODO ([0, 0.7 * max(t_loss)]) ???
    plt.grid(axis = 'y')
    # plt.xticks(range(epochs))

    # plt.show()
    plt.savefig(run_folder+'/Point_Learning_Curve.png', bbox_inches='tight')
    plt.close()
    
# %%
#################################################################
# plot mse distribtion for all images
def plot_MSE_distribution(run_folder, points_df):

    mse = np.vstack(points_df.pt_mse_loss)
    x_low = mse.mean() - mse.std()
    x_mean = mse.mean()
    x_hi  = mse.mean() + mse.std()
    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparam from run_folder

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle('Point MSE Loss - ' + title_str, fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    ax.set_title(f'MSE mean={mse.mean():1.4f}  std={mse.std():1.4f}  min={mse.min():1.4f}  max={mse.max():1.4f} for {mse.shape[0]} samples')

    plt.hist(mse, bins=50, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(x_low, color='g', linestyle='dotted', linewidth=1) 
    plt.axvline(x_mean, color='k', linestyle='solid', linewidth=1) 
    plt.axvline(x_hi,  color='r', linestyle='dotted', linewidth=1)
    # plt.grid()
    # plt.xlim(0, 0.40)
    # plt.ylim(0, 1000)
    plt.xlabel("MSE with Mean +/- Std")
    plt.ylabel("Number of Images")

    # plt.show()
    plt.savefig(run_folder+'/Point_MSE_Loss.png')
    plt.close()

# %%
#################################################################
# plot std distribtion for all images
def plot_STD_distribution(run_folder, points_df):       
    # >>>> TODO boxplot instead of hist 
    # https://matplotlib.org/stable/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py

    std = np.vstack(points_df.pt_std)
    n_samples = std.shape[0]
    n_dims = std.shape[1]
    std = std.flatten()
    x_low = std.mean() - std.std()
    x_mean = std.mean()
    x_hi  = std.mean() + std.std()
    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparam from run_folder

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle('Point Fuzziness - ' + title_str, fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    ax.set_title(f'STD mean={std.mean():1.4f}  std={std.std():1.4f}  min={std.min():1.4f}  max={std.max():1.4f} for {n_samples} samples')

    plt.hist(std, bins=50, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(x_low, color='g', linestyle='dotted', linewidth=1) 
    plt.axvline(x_mean, color='k', linestyle='solid', linewidth=1) 
    plt.axvline(x_hi,  color='r', linestyle='dotted', linewidth=1)
    # plt.grid()
    # plt.xlim(0, 0.40)
    # plt.ylim(0, 1000)
    plt.xlabel(f"Point STD across all {n_dims} dims with Mean +/- one Std")
    plt.ylabel(f"Number of Point Positions ({n_samples} samples * {n_dims} dims)")

    # plt.show()
    plt.savefig(run_folder+'/Point_STD_Fuzziness.png')
    plt.close()

# %%
#################################################################
# Plot distribution of each latent dim 
def plot_latent_space_density(run_folder, points_df):

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder

    nBins = 100 # plus one
    pos = np.vstack(points_df.pt_encoded)
    lab = np.vstack(points_df.pt_label)

    nPoints = pos.shape[0]
    nDim = pos.shape[1]
    pos_norm = nBins * (pos - pos.min()) / (pos.max() - pos.min())
    pos_bins = (pos_norm + 0.5).astype(int)

    density = np.zeros((nDim, nBins+1), dtype=np.int32)
    for i in range(nPoints):
        for j in range(nDim):
            k = pos_bins[i,j]
            density[j, k] += 1

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle(f'L-Space Density -- {title_str}', fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    ax.set_title(f'Density mean={density.mean():.1f}  min={density.min():d}  max={density.max():d}')

    plt.imshow(density, cmap ='Greens', aspect='auto')
    plt.colorbar()
    plt.grid()
    plt.xlabel("Point Latent Position")
    plt.ylabel("Point Dimensions")
    
    # set x-axis ticks with pos.min to pos.max
    x_pos = np.arange(0,101,20)
    x_labels = [str(round(x, 1)) for x in np.linspace(pos.min(), pos.max(), num=6)]
    plt.xticks(x_pos,x_labels)

    # plt.show()
    plt.savefig(run_folder+'/L_Space_Density.png')
    plt.close()

# %%
#################################################################
# Plot class entanglement across all latent dims
def plot_latent_space_entangle(run_folder, points_df):

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder

    pos = np.vstack(points_df.pt_encoded)
    lab = np.vstack(points_df.pt_label)
    lab_unique = np.unique(lab)
    # labstr = np.vstack(points_df.pt_labstr)     # >>>> TODO handle conversions label <=> label-str
    # labstr_unique = np.unique(labstr)
    n_dim = pos.shape[1]

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle(f'L-Space Entanglement -- {title_str}', fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    # ax.set_title(f'Density mean={density.mean():.1f}  min={density.min():d}  max={density.max():d}')

    dat = []
    for c in lab_unique:
        x = [np.mean(pos[lab[:, 0] == c, i], axis=0) for i in range(n_dim)]
        dat.append([c] + x)
        y = [n_dim - i - 1 for i in range(n_dim)]
        scat = plt.plot(x, y)
    
    plt.grid()
    plt.xlabel("Point Latent Position (normalized)")
    plt.ylabel("Point Dimensions")
    
    # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scat.legend_elements(),
    #                     loc="upper left", title="Classes")
    # ax.add_artist(legend1)
    
    # set x-axis ticks with pos.min to pos.max
    # x_pos = np.arange(0,101,20)
    # x_labels = [str(round(x, 1)) for x in np.linspace(pos.min(), pos.max(), num=6)]
    # plt.xticks(x_pos,x_labels)

    # plt.show()
    plt.savefig(run_folder+'/L_Space_Entanglement.png')
    plt.close()
    
    # create LS_entangle dataframe
    col = ['label'] + ['dim'+str(d) for d in range(n_dim)]
    LS_entangle_df = pd.DataFrame(data=dat, columns=col)
    
    return LS_entangle_df

# %%
#################################################################
# plot first N images with their reconstruction
import math, random
def plot_reconstructed_images(  run_folder, 
                                points_df, 
                                id_key,         # if list, plot keyed images; else tuple (plot_type, n_samples)
                                                # where type_samples = 'first', 'random', 'lo-mse', 'hi-mse'
                                tag='', 
                                binarize=True,  # convert decoded_data to 0.0 or 1.0 at 0.5 cutoff
                                save_plot=True
                            ):

    N_MAX = 36
    N_PER_ROW = 6

    # setup id_key param for id_lst
    if type(id_key) is list:
        id_lst = id_key
        n_img = len(id_lst)
    elif type(id_key) is tuple:
        plot_type, n_img = id_key
        tag = plot_type.title() + ' ' + str(n_img)
        if plot_type.lower() == 'first':
            id_lst = range(n_img)
        elif plot_type.lower() == 'last':
            id_lst = range(0, n_img, -1)
        elif plot_type.lower() == 'random':
            id_lst = random.sample(range(len(points_df.index)), k=n_img)
        elif plot_type.lower() == 'lo-mse':
            id_lst = np.argsort(points_df['pt_mse_loss'].to_numpy())[:N_MAX]
        elif plot_type.lower() == 'hi-mse':
            id_lst = np.argsort(points_df['pt_mse_loss'].to_numpy())[::-1][:N_MAX]
        else: print(f'    ERROR: Bad id_key param "{id_key}" with unknown plot_type')
    else: print(f'    ERROR: Bad id_key param "{id_key}" as not list or tuple')

    n_img = min(n_img, N_MAX)
    n_rows = int(math.ceil(n_img / N_PER_ROW))

    # get data from points_df
    feat_data = np.array(points_df['pt_feature'].tolist())
    decoded_data = np.array(points_df['pt_decoded'].tolist())
    if binarize:
        decoded_data = np.where(decoded_data > 0.5, 1, 0)
    mse = points_df['pt_mse_loss'].to_numpy()
    s_img = int(math.sqrt(feat_data[0,:].shape[0])) # get square height/width size

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle(f'Reconstructed Points ({tag}) -- {title_str}', fontsize=12, fontweight="bold")
    ax = fig.add_subplot()
    ax.set_title(f'First image original, second recontructed, title with sample id + pre-label')
    
    for i, id in enumerate(id_lst): 

        img1 = feat_data[id, :].reshape((s_img,s_img))
        img2 = decoded_data[id, :].reshape((s_img,s_img))
        img = np.hstack((img1, img2))

        ax = plt.subplot(n_rows, N_PER_ROW, i + 1)
        # plt.imshow(img, cmap='binary_r', aspect='equal')
        plt.imshow(img, cmap='binary_r')

        labstr = points_df['pt_labstr'].iloc[id] 
        ax_title = f'#{str(id)} - {labstr.upper()}'
        ax.set_title(ax_title, fontsize=8, pad=0)
        ax.axvline(s_img, color='k', linestyle='solid', linewidth=1)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i+1 >= n_img: break

    if save_plot:
        plt.savefig(f'{run_folder}/Point_Reconstructed_Samples{"_" + tag}.png')
    else:
        plt.show()
    plt.close()

# %%
######### load a previous points_df to avoid retraining the same latent space
def load_previous_points(points_path):

    # points_df = pd.read_pickle(points_path + "/points_df.pkl")
    points_df = pd.read_json(points_path + "/points_df.json")
    return points_df

# %%
######### Preprocess dataset images into NN data vectors

def create_points(  samples_df, 
                    image_size=32,          # 32, 64, 96, 128
                    # preprocess=True,       # whether to preprocess images with smart-crop   >>> TODO remove! samples_df is responsible
                    nDim=16,                # latent dims 2...
                    nEpochs=10,             # num of epochs in training cycle
                    # loss_reduction='mean',  # 'mean' vs 'sum'                 TODO future... add param
                    run_folder='./',        # where to save df and png files
                    verbose=True,           # print log to file
                ):
    '''
    
    '''
    ##### train model using latent_dim hparam
    if verbose: 
        print(f'>>> CREATING POINTS with epochs={nEpochs}, latent_dim={nDim}, ')

    # custom transforms for creating torch DataSet >>> TODO where did this coming from? Use 'None' for now.
    # my_transforms = torchvision.transforms.Compose([
    #                 torchvision.transforms.ToTensor(),
    #                 torchvision.transforms.Resize(image_size),
    #                 torchvision.transforms.CenterCrop(image_size),
    #                 ])
    
    # instantiate torch custom DataSet
    # train_dataset = VAE_Train_Dataset(samples_df, preprocess=preprocess, transform=None)  >>> TODO remove!!! samples_df responsible
    train_dataset = VAE_Train_Dataset(samples_df, transform=None)

    # instantiate training DataLoader
    batch_size = len(samples_df.index) // 100   # set # of batch to roughly 100
    train_loader = DataLoader(  dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                # drop_last=True,
                                # sampler=train_sampler, 
                            )
    
    # check sample shape from dataloader
    img, _ = next(iter(train_loader))
    if verbose:
        print(f'    Feature batch shape = {img.shape} with DEVICE = {DEVICE}')

    num_channels, img_height, img_width = img.shape[1:4]

    if img_height != img_width: # must be square
        print(f'    ERROR: image is NOT square')
    img_size = img_height
    if not (img_size % 32 == 0) & (img_size >= 32) & (img_size <= 256):
        print(f'    ERROR: image size is NOT mutiple  of 32 and in 32..256')

    # instantiate model and print structure
    model = VAE(num_channels, img_size, nDim)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    log_dict = train_VAE(   num_epochs=nEpochs, model=model, 
                            optimizer=optimizer, device=DEVICE, 
                            train_loader=train_loader,
                            skip_epoch_stats=True,      # TODO False => only log epoch stats???
                            beta=1,                     # TODO try higher values for beta 2...10 affect on reconstruction
                            logging_interval=50,
                            save_model=run_folder + '/VAE_MNIST_Roman.pt'
                        )

    # instantiate encode-decode DataLoader, same order as original dataset      TODO check whether images are sync with img_id !!!
    encode_decode_loader = DataLoader(  dataset=train_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                    )
    feat_lst=[]; encoded_lst=[]; z_mean_lst=[]; z_log_var_lst=[]; decoded_lst=[]; mse_lst=[]

    with torch.no_grad():   # turn off gradient 
        model.eval()

        for idx, (features, _) in enumerate(encode_decode_loader):
            # forward prop
            features = features.to(DEVICE)
            encoded, z_mean, z_log_var, decoded = model(features)
            # decoded2 = model.decoder(model.z_mean(model.encoder(features)))   # TODO does decoded == decoded2?
            
            feat_lst.append(features.cpu().detach().numpy().flatten())
            encoded_lst.append(encoded.cpu().detach().numpy().flatten())
            z_mean_lst.append(z_mean.cpu().detach().numpy().flatten())
            z_log_var_lst.append(z_log_var.cpu().detach().numpy().flatten())
            decoded_lst.append(decoded.cpu().detach().numpy().flatten())

            loss = F.mse_loss(decoded, features, reduction='mean')   # >>> TODO Why 'sum' versus 'mean'?
            mse_lst.append(loss.cpu().detach().numpy().flatten()) 

    num_samples = len(encoded_lst)
    assert num_samples == len(train_dataset)    # should be equal to len(train_dataset)
    feat_data = np.array(feat_lst)
    encoded_data = np.array(encoded_lst)
    z_mean_data = np.array(z_mean_lst)
    z_log_var_data = np.array(z_log_var_lst)
    z_std_data = np.exp(0.5 * z_log_var_data)   # convert log of var to std deviation

    decoded_data = np.array(decoded_lst)

    mse_loss = np.array(mse_lst)
    mse_mean = float(mse_loss.mean())
    mse_std = float(mse_loss.std())

    # ##### create/save points_df table keyed to samples_df
    points_df = pd.DataFrame({'id': samples_df.index})
    points_df['flags'] = ''
    points_df['pt_label'] = [l for l in samples_df['img_label']]    # just copy from samples
    points_df['pt_labstr'] = [c for c in samples_df['img_labstr']]  # just copy from samples
    points_df['pt_feature'] = [pix for pix in feat_data]
    points_df['pt_encoded'] = [pos for pos in encoded_data]
    points_df['pt_decoded'] = [pix for pix in decoded_data]
    points_df['pt_pos'] = [pos for pos in z_mean_data]          # >>>>> TODO investigate pt_encoded == pt_pos ???
    points_df['pt_std'] = [std for std in z_std_data]
    points_df['pt_mse_loss'] = mse_loss

    # print/plot results
    if verbose: 

        print(f'>>> Feature/Encode/Decode shapes = {feat_data.shape} {encoded_data.shape} {decoded_data.shape}')
        print(f'>>> Z_mean/Z_logvar/Z_std shapes = {z_mean_data.shape} {z_log_var_data.shape} {z_std_data.shape}')
        print(f'>>> MSE shape = {mse_loss.shape} mean={mse_mean:0.4f} std={mse_std:0.4f} ' + 
                f'min={mse_loss.min():0.4f} max={mse_loss.max():0.4f}')
        mse_z = (mse_loss - mse_loss.mean()) / mse_loss.std()
        mse_out = mse_loss[mse_z > 3]
        pct = len(mse_out) / len(mse_loss)
        print(f'>>> MSE outliers: {len(mse_out):,d} or {pct:.1%} with Z > 3')        
        # check that plt.imshow images of samples same as points                >>>> TODO 
        # check_sample_to_point_images(samples.img_array, points.pt_feature)
        
        # plot learning curves
        plot_learning_curve(run_folder, log_dict, nEpochs, num_samples)
        
        # plot MSE distribution
        plot_MSE_distribution(run_folder, points_df)
        
        # plot pos_std distribution
        plot_STD_distribution(run_folder, points_df)
        
        # plot density of latent space
        plot_latent_space_density(run_folder, points_df)
        
        # plot entanglement of latent space 
        LS_entangle_df = plot_latent_space_entangle(run_folder, points_df)
        # save_dataframe(run_folder, 'LS_entangle_df', LS_entangle_df)
        
        # plot original vs reconstruction images for First, Lo-MSE, Hi-MSE, etc
        plot_reconstructed_images(run_folder, points_df, ('First', 36), binarize=True)
        plot_reconstructed_images(run_folder, points_df, ('Lo-MSE', 36), binarize=True)
        plot_reconstructed_images(run_folder, points_df, ('Hi-MSE', 36), binarize=True)

    return points_df

# %%
# Use UMAP to fit 8D LS samples to 2D and 3D space
def add_posLowD_to_points(points_df, nNeighbers, minDist):

    pos = np.stack(points_df['pt_encoded'])     # >>>> TODO use pt_pos instead? Make a difference?
    
    for d in [2, 3]:
        umap_object = umap.UMAP(
            n_neighbors=nNeighbers,     # set from hyperparam
            min_dist=minDist,           # set from hyperparam
            n_components=d,             # set to 2D
            random_state=42)
        umap_object.fit(pos)

        points_df[f'pt_pos{d}D'] = [pos for pos in umap_object.transform(pos)]
            
    return points_df

# >>>>> TODO random note! where does this go?
# Use UMAP to create graph object for transforming lS and edge creation
#   ...for enhanced clustering re https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering

# %%
# Use UMAP to fit 8D LS samples to 3D space (which is used in LS_workshop)

def plot_pos2D_points(points_df, nNeighbers, minDist, run_folder):

    pos2D = np.stack(points_df['pt_pos2D'])
    pt_marker = '.' if pos2D.shape[0] > 10_000 else 'o' # adjust marker depending on # of points
    lab = np.stack(points_df['pt_label'])
    # labstr = np.stack(points_df['pt_labstr'])         # TODO add below to legend

    # umap_object = umap.UMAP(              >>>> redundant with create_2D_points
    #     n_neighbors=nNeighbers,     # set from hyperparam
    #     min_dist=minDist,           # set from hyperparam
    #     n_components=2,             # set to 2D
    #     random_state=42)
    # umap_object.fit(pos)
    # pos2D = umap_object.transform(pos)

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder
    # plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    plt.title(f'2D Latent Space --' + title_str, fontsize=12, fontweight='bold')
    # ax = fig.add_subplot(projection='3d')
    scat = ax.scatter(pos2D[:,0], pos2D[:,1], c=lab, marker=pt_marker, s=3, alpha=1.0, cmap='tab10', edgecolors='none')

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scat.legend_elements(),
                        loc="upper left", title="Classes")
    ax.add_artist(legend1)

    # plt.show()
    plt.savefig(run_folder+'/2D_Latent_Space.png')
    plt.close()

    return points_df

