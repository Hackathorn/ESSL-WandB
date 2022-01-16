# %%
# import modules

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
from glob import glob
from torch.utils import data
from torchvision import datasets, transforms

# %%
# set global params
DATA_PATH = '/data'     # path to structured folders containing datasets

# # %%
# # test whether Image.img is black-on-white (preferred) or white-on-black (invert!)
# def is_white_on_black(img):
    
#     # convert to np.array & select only the top & left edges
#     im = np.array(img)[:3, :3]
#     # then min-max normalize to [0...1]
#     im_norm = (im - min(im)) / (max(im) - min(im))
    
#     # test for high average
#     return np.average(im_norm) > 0.5
    
# %%
# Process a single PNG Image

def process_image(img,              # input Image.image
                mode='L',           # check image.mode; only dealing with gray scale currently
                invert='BonW',      # invert image to white-on-black (WonB), black-on-white (BonW) or None ()
                crop=False,         # crop w/h to bbox
                square=False,       # square image so that w = h
                cutoff=None,        # binarize pixels to 0 or 255
                resize=None,        # resize to tuple (width, height)
                border=0,           # expand image edge by border with 0's
                normalize=False,    # divide by 255 and change np type to float32
                ):
    """Processes single Image.image with mode check, invert, crop, square, pixel cutoff binarize, resize, and border

    Returns:
        [type]: [description]
    """
    # Is mode == 'L'?  
    if img.mode != mode:
        print(f'WARNING: Image mode is not {mode} for PNG image')
    
    # # invert image ...if invert is not None AND image is not WonB or BonW as desired
    # if (invert is not None):
    #     if invert == 'WonB' and not is_white_on_black(img):
    #         img = ImageOps.invert(img)
    #     elif invert == 'BonW' and is_white_on_black(img):
    #         img = ImageOps.invert(img)
    #     else: 
    #         raise ValueError(f'Bad value {invert} for invert param')
    # img = ImageOps.invert(img)  # >>>>> TODO always inverting img
            
    # Crop image to bounding box
    if crop:
        img = img.crop(img.getbbox())       # >>>> TODO check that background is black
    
    # paste into square image, preserving aspect ratio
    if square:
        (w, h) = img.size
        new_size = (w, w) if w > h else (h, h)
        new_img = Image.new('L', new_size, color=0)	# create image and initialize to background
        new_box  = (0, (w-h)//2) if w > h else ((h-w)//2, 0)
        new_img.paste(img, new_box)
        img = new_img
    
    # add border
    if border > 0:
        img = ImageOps.expand(new_img, border=border, fill=0)   # TODO check background pixel value

    # Resize if required
    if resize != None:
        if square & (resize[0] != resize[1]):
            print('WARNING: Unsquaring a squared image with resize!')
            resize = (resize[0], resize[0]) if resize[0] > resize[1] else (resize[1], resize[1])
            print(f'WARNING: Squaring image to {resize}')
        img = img.resize(resize, Image.BICUBIC)  # Image.ANTIALIAS?


    # binarize pixel values at cutoff
    if cutoff != None:
        img_array = np.array(img)
        img_array[img_array > cutoff] = 255
        img_array[img_array <= cutoff] = 0
        img = Image.fromarray(img_array)

    # standarize pixel values to [0,1] and convert to float32 for TF-Keras & pyTorch
    if normalize:
        # img = np.expand_dims(img, -1) / np.max(np.array(img))   # Note: could be <255
        img = np.array(img) / 255
        
    return np.array(img, dtype=np.float32)    # return array instead of Image

# %%
# Make dataset from CheckerBoard patterns
def make_checker_images(
    n_samples=100, 
    n_classes=10,
    *,
    n_img_size=32, 
    n_blk_size=2, 
    shuffle=True, 
    random_state=None, 
    return_class_patterns=False 
):
    """Make 2D images with random checkerboard patterns
    
    Parameters:
        n_samples : int, default=100    
            The number of sample 2D images
        n_classes : int, default=10     >>> TODO accept tuple for multi-classes
            The number of classes (as unique bit patterns) for sample images
        n_img_size : int, default=32    >>> TODO accept tuple for non-square images
            The width and height in pixels of square image. Values=32,64,128
        n_blk_size : int, default=1, possible=1,2,3... to 1/2 of n_img_size    
            The width/height of pixel blocks to control information density 
        shuffle : bool, default=True
            Shuffle the samples
        random_state : int or None, default=None
            Determines random number generation. Pass int for reproducible results
        return_class_patterns : bool, default=False 
            If True, also returns list of bit patterns for each class as str of 0/1
        
    Returns:
        X : ndarray of shape (n_samples, n_img_size, n_img_size)
            samples
        y : ndarray of shape (n_samples)
            targets
        patterns : list of str, len=n_classes
            list of bit patterns for each class as string of 0/1's
    """

    rng = np.random.default_rng(random_state)

    # get total no of pixel blocks per square image
    ww = n_img_size // n_blk_size     # no of pixel blocks per row or column
    n_bks = ww * ww

    # generate random 0/1 for each pixel block across all classes  >>> TODO check for duplicate patterns
    class_bits = [rng.integers(0, 2, n_bks) for k in range(n_classes)]
    class_patterns = [''.join([str(b) for b in class_bits[k][:]]) for k in range(n_classes) ]
    
    if n_classes > n_bks: 
        print(f'ERROR: n_classes={n_classes} is too big for {n_bks} pixel blocks. Reduce n_blk_size param. ')

    # convert class_bits into 1/0 array of shape(n_classes, n_img_size, n_img_size)
    img_patterns = np.zeros((n_classes, n_img_size, n_img_size))

    # cycle thru each class for its random pattern
    for k, bits in enumerate(class_bits):
        # cycle thru each pattern bits for that class
        for nb, b in enumerate(bits):
            i = nb // ww; j = nb % ww
            for d1 in range(n_blk_size):         # cycle pixel block thru next column 
                for d2 in range(n_blk_size):     # cycle pixel block thru next row 
                    ii = n_blk_size*i + d1; jj = n_blk_size*j + d2
                    img_patterns[k, ii, jj] = b

    # create dataset of N random samples across n_classes classes
    n_class_samples = (n_samples // n_classes)  # integer floor-division
    n_samples2 = n_classes * n_class_samples    # revise samples down if needed
    samples = np.zeros((n_samples2, n_img_size, n_img_size), dtype=np.float32)
    targets = np.zeros((n_samples2), dtype=np.float32)
    
    # generate samples for n_classes, truncating n_samples if needed
    for i in range(n_samples2):
        k = i // n_class_samples    # track class index
        samples[i, :] = img_patterns[k, :]
        targets[i] = k
    
    # randomly shuffle samples/targets arrays if requested
    if shuffle:
        indices = np.arange(n_samples2)
        rng.shuffle(indices)
        samples = samples[indices]
        targets = targets[indices]

    # return results    
    if return_class_patterns:
        return samples, targets, class_patterns
    else:
        return samples, targets

# %%
# Find dataset folder in DATA_PATH directory
def find_dataset_folder(dataset_name):
    pass  # TODO for generic datasets by create_samples_from_folder
    
# %%
def create_samples_from_DCAI(dataset_name, img_size, img_invert=False, verbose=True): # >>>>> TODO refactor into ONE create_samples
    """create_samples from dataset returning samples_df with image attributes

    Args:
        dataset_name ([type]): [description]
        img_size ([type]): [description]
        img_invert (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]

    Assumes folder structure of: 
        status_folders (train, val, book, delete, new...) 
            class_folders (i, ii, iii...) that map to labels [0..L] with label_to_labstr list 
                PNG image files with mode='L' (one-byte grayscale)
    """
    assert dataset_name == 'DCAI'    # should be DCAI dataset as a local 'structured' folder

    # NOTE: can lower-case by... [c.lower() for c in label_to_labstr]    >>>> TODO add in future CLASS def
    # label_to_labstr = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    label_to_labstr = ['0-I', '1-II', '2-III', '3-IV', '4-V', '5-VI', '6-VII', '7-VIII', '8-IX', '9-X']
    sample_list = []
    flags = ''
    resize = (img_size, img_size)    

    status_folders = glob(os.path.join(DATA_PATH + dataset_name + '\\', '*'))   # find all status_folders (train, val...)
    for status_folder in status_folders:

        lab_folders = glob(os.path.join(status_folder, '*'))    # find all label folders (i, ii...)
        for lab_folder in lab_folders:

            files = glob(os.path.join(lab_folder, "*.png"))     # find all PNG file in that label folder
            for id, file_path in enumerate(files):
                
                img = Image.open(file_path)								# open (but do not invert) each image
                if img_invert:
                    img = ImageOps.invert(img)

                # get bounding box of image
                img_bbox = img.getbbox()
                # if img_bbox == (0, 0, img.width, img.height):	  >>>>>>>>>>>> TODO: debug
                # 	print('WARNING: Inverting image for BBOX')
                # 	img_bbox = ImageOps.invert(img).getbbox()	# assume image invert is needed

                # process image from Image.open above
                img_out = process_image(img, 
                    mode='L', 
                    invert='BonW', 
                    crop=True, 
                    square=True, 
                    cutoff=50, 
                    resize=resize, 
                    border=2,
                    normalize=True),    # TODO True ???
                
                if verbose and (id == 0): 
                    print(f'    Image shape = {"img_out.shape"} of type = {type(img_out)}')
                # add label/class columns
                img_labstr = file_path.split('\\')[4]   # str name of class folder [i, ii...x] from path split
                img_label = label_to_labstr.index(img_labstr.upper())

                # save image attributes into samples_df; note that arrays are wrapped as list type
                sample_list.append([id, file_path, flags, img.width, img.height, img_bbox, img.mode, [img_out], img_label, img_labstr])

    # create dataframe for samples
    samples_df = pd.DataFrame(sample_list, columns=['id', 'file_path', 'flags', 'img_width', 'img_height', 
                                                    'img_bbox', 'img_mode', 'img_array', 'img_label', 'img_labstr']) 
    samples_df = samples_df.astype({'file_path': 'string', 'flags': 'string', 'img_mode': 'string', 'img_labstr': 'string'})
    samples_df = samples_df.astype({'img_width': np.int16, 'img_height': np.int16, 'img_label': np.int16})

    return samples_df

# %%
# Load MNIST dataset from TorchVision to create samples_df
def create_samples_from_MNIST(ds_name, img_size, img_invert=True, verbose=True):
    """create_samples from MNIST dataset returning samples_df with image attributes

    Args:
        ds_name ([type]): [description]
        img_size ([type]): [description]
        img_invert (bool, optional): [description]. Defaults to False since 

    Returns:
        [type]: [description]
    """
    if ds_name == 'MNIST':           # >>>> TODO add EMNIST, KMNIST, QMNEST from TorchVision
        ds = datasets.MNIST(root='./data', train=True, download=True)
        label_to_labstr = ['0-zero', '1-one', '2-two', '3-three', '4-four', '5-five', 
                           '6-six', '7-seven', '8-eight', '9-nine']
    elif ds_name == 'FASHION':
        ds = datasets.FashionMNIST(root='./data', train=True, download=True)
        label_to_labstr = ["0-TShirt", "1-Trouser", "2-Pullover", "3-Dress","4-Coat","5-Sandal", 
                           "6-Shirt", "7-Sneaker", "8-Bag", "9-Ankle Boot",]
    else:
        print(f'ERROR: Invalid MNIST dataset {ds_name}')
        return
    
    sample_list = []
    flags = ''
    resize = (img_size, img_size)    
    
    for id, (img, img_label) in enumerate(ds):

        if img_invert:
            img = ImageOps.invert(img)

        # get bounding box of image
        img_bbox = img.getbbox()
        # if img_bbox == (0, 0, img.width, img.height):	  >>>>>>>>>>>> TODO: debug
        # 	print('WARNING: Inverting image for BBOX')
        # 	img_bbox = ImageOps.invert(img).getbbox()	# assume image invert is needed

        # process image from Image.open above
        img_out = process_image(img, 
            mode='L', 
            invert='BonW', 
            crop=True, 
            square=True, 
            cutoff=50, 
            resize=resize,              # pad border by 2
            border=2,
            normalize=True),    # TODO True ???
        
        # if verbose and (id == 0): 
        #     print(f'    Image shape = {img_out.shape} of type = {type(img_out)}')

        # add labstr column from label_to_labstr map
        img_labstr = label_to_labstr[img_label]

        # save image attributes into samples_df
        sample_list.append([id, ds_name, flags, img.width, img.height, img_bbox, img.mode, [img_out], img_label, img_labstr])
        
    # create dataframe for samples
    samples_df = pd.DataFrame(sample_list, columns=['id', 'ds_name', 'flags', 'img_width', 'img_height', 
                                                    'img_bbox', 'img_mode', 'img_array', 'img_label', 'img_labstr']) 
    samples_df = samples_df.astype({'ds_name': 'string', 'flags': 'string', 'img_mode': 'string', 'img_labstr': 'string'})
    samples_df = samples_df.astype({'img_width': np.int16, 'img_height': np.int16, 'img_label': np.int16})

    return samples_df

# %%
# Load random checkerboard dataset to create samples_df
def create_samples_from_CHECKER(ds_name, img_size, n_classes=10, n_blk_size=2, img_invert=False, verbose=True):
    """create_samples from CHECKER dataset returning samples_df with image attributes

    Args:
        dataset_name ([type]): [description]
        img_size ([type]): [description]
        img_invert (bool, optional): [description]. Defaults to False since 

    Returns:
        [type]: [description]
    """
    assert ds_name == 'CHECKER', "should be CHECKER dataset"
    
    X, y = make_checker_images(3000, n_classes, n_img_size=img_size, n_blk_size=n_blk_size)
    
    label_to_labstr = [f'pat{k:02d}' for k in range(n_classes)]
    flags = ''
    
    sample_list = []
    for id, img in enumerate(X[:]):
        
        img_label = y[id]

        # if img_invert:
        #     img = ImageOps.invert(img)

        # get bounding box of image
        img_bbox = (0, 0, img_size, img_size)
        
        # img_bbox = img.getbbox()
        # if img_bbox == (0, 0, img.width, img.height):	  >>>>>>>>>>>> TODO: debug
        # 	print('WARNING: Inverting image for BBOX')
        # 	img_bbox = ImageOps.invert(img).getbbox()	# assume image invert is needed

        # process image from Image.open above
        img_out = np.array(X[id,:], dtype=np.float32)
        
        # img_out = process_image(img, 
        #     mode='L', 
        #     invert='BonW', 
        #     crop=True, 
        #     square=True, 
        #     cutoff=50, 
        #     resize=resize, 
        #     border=2,
        #     normalize=True),    # TODO True ???
        
        # if verbose and (id == 0): 
        #     print(f'    Image shape = {img_out.shape} of type = {type(img_out)}')

        # add labstr column from label_to_labstr map
        img_labstr = label_to_labstr[int(img_label)]

        # save image attributes into samples_df
        sample_list.append([id, ds_name, flags, img_size, img_size, img_bbox, 'L', [img_out], img_label, img_labstr])
        
    # create dataframe for samples
    samples_df = pd.DataFrame(sample_list, columns=['id', 'ds_name', 'flags', 'img_width', 'img_height', 
                                                    'img_bbox', 'img_mode', 'img_array', 'img_label', 'img_labstr']) 
    samples_df = samples_df.astype({'ds_name': 'string', 'flags': 'string', 'img_mode': 'string', 'img_labstr': 'string'})
    samples_df = samples_df.astype({'img_width': np.int16, 'img_height': np.int16, 'img_label': np.int16})

    return samples_df

# %%
# Create samples but first determine which dataset  # >>>>> TODO refactor into ONE create_samples

def create_samples(ds_name, img_size, n_classes=10, n_blk_size=2, verbose=True):
    
    if verbose: 
        print(f'>>> CREATING SAMPLES for {ds_name} with img_size = {img_size}')
    
    if ds_name in ['MNIST', 'FASHION']:
        sample_df = create_samples_from_MNIST(ds_name, img_size, verbose=verbose)
    elif ds_name == 'DCAI':
        sample_df = create_samples_from_DCAI(ds_name, img_size, verbose=verbose)
    elif ds_name == 'CHECKER':
        sample_df = create_samples_from_CHECKER(ds_name, img_size, n_classes=n_classes, 
                                                n_blk_size=n_blk_size, verbose=verbose)
    # elif:                       # >>>>> TODO add later for any dataset in data folder
    #     sample_df = create_samples_from_folder(dataset_name, img_size, img_invert=img_invert)
    else:
        raise ValueError(f'Bad value for dataset_name = "{ds_name}"')
        
    return sample_df
