# %%
import sys
import os
import time
import numpy as np
import math
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..') # the path containing "suns" folder

# from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.PreProcessing.generate_masks import generate_masks


# %%
if __name__ == '__main__':
    # %% setting parameters
    rate_hz = 10 # frame rate of the video
    Dimens = (120,88) # lateral dimensions of the video
    nn = 3000 # number of frames used for preprocessing. 
        # Can be slightly larger than the number of frames of a video
    num_total = 3000 # number of frames used for CNN training. 
        # Can be slightly smaller than the number of frames of a video
    Mag = 6/8 # spatial magnification compared to ABO videos.

    thred_std = 3 # SNR threshold used to determine when neurons are active

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
        # Can only be used when spatial filtering is not used. 
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = 'data\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 
    dir_parent = dir_video + 'complete\\' # folder to save all the processed data
    dir_network_input = dir_parent + 'network_input\\' # folder of the SNR videos
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks

    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the original video
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = 'YST_spike_tempolate.h5'

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # # Alternative temporal filter kernel using a single exponential decay function
    # decay = 0.8 # decay time constant (unit: second)
    # leng_tf = np.ceil(rate_hz*decay)+1
    # Poisson_filt = np.exp(-np.arange(leng_tf)/rate_hz/decay)
    # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')

    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    # pre-processing for training
    for Exp_ID in list_Exp_ID[0:1]: #
        # %% Pre-process video
        # video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
        #     useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #
        h5_img = h5py.File(dir_network_input+Exp_ID+'.h5', 'r')
        video_input = np.array(h5_img['network_input'])

        # %% Determine active neurons in all frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
        # del video_input

