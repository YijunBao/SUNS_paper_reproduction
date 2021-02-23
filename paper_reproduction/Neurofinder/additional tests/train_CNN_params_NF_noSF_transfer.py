# %%
import sys
import os
import random
import time
import glob
import numpy as np
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp
from shutil import copyfile

sys.path.insert(1, '..\\..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.PreProcessing.generate_masks import generate_masks
from suns.train_CNN_params import train_CNN, parameter_optimization_cross_validation


# %%
if __name__ == '__main__':
    # %% setting parameters
    list_neurofinder_train = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    list_neurofinder_test = [x+'.test' for x in list_neurofinder_train]
    px_um = [1/0.8, 1/0.8, 1/1.15, 1/1.15, 0.8, 1.25]
    list_Mag = [x*0.78 for x in px_um]
    list_rate_hz = [7.5, 7.5, 8, 8, 6.75, 3] # [3] * 6 # 
    Dimens = [(504,504), (504,504), (464,504), (464,504), (416,480), (416,480)] # lateral dimension of the video
    list_nframes_train = [2250, 1825, 8000, 8000, 3000, 3000]
    list_nframes_test = [2250, 5000, 8000, 8000, 3000, 3000]

    thred_std = 3 # SNR threshold used to determine when neurons are active
    num_train_per = 1800 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 100 # batch size in CNN inference

    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.
    useWT=False # True if using additional watershed
    load_exist=False # True if using temp files already saved in the folders
    use_validation = False # True to use a validation set outside the training set
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "train_1_test_rest"
    Params_loss = {'DL':1, 'BCE':1, 'FL':0, 'gamma':1, 'alpha':0.25} # Parameters of the loss function

    for trainset_type in {'train', 'test'}: # 
        # valset_type = list({'train','test'}-{trainset_type})[0]
        # %% set folders
        if trainset_type == 'train':
            list_nframes = list_nframes_train
            list_Exp_ID = list_neurofinder_train
            # list_Exp_ID_val = list_neurofinder_test
        else: # if trainset_type == 'test':
            list_nframes = list_nframes_test
            list_Exp_ID = list_neurofinder_test
            # list_Exp_ID_val = list_neurofinder_train

        # folder of the raw videos
        dir_video = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
        dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 
        dir_parent = dir_video + 'noSF\\' # folder to save all the processed data
        dir_network_input = dir_parent + 'network_input\\' # folder of the SNR videos
        dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks
        dir_sub = 'transfer1e-4\\'
        weights_path = dir_parent + dir_sub + 'Weights\\' # folder to save the trained CNN
        training_output_path = dir_parent + dir_sub + 'training output\\' # folder to save the loss functions during training
        dir_output = dir_parent + dir_sub + 'output_masks\\' # folder to save the optimized hyper-parameters
        dir_temp = dir_parent + dir_sub + 'temp\\' # temporary folder to save the F1 with various hyper-parameters
        exist_model = 'D:\\ABO\\20 percent\\noSF\\Weights\\Model_CV10.h5'

        if not os.path.exists(dir_network_input):
            os.makedirs(dir_network_input) 
        if not os.path.exists(weights_path):
            os.makedirs(weights_path) 
        if not os.path.exists(training_output_path):
            os.makedirs(training_output_path) 
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp) 

        # nvideo = len(list_Exp_ID) # number of videos used for cross validation

        for (ind_video, Exp_ID) in enumerate(list_Exp_ID): # 
            rate_hz = list_rate_hz[ind_video] # frame rate of the video
            nframes = list_nframes[ind_video] # number of frames for each video
            Mag = list_Mag[ind_video] # spatial magnification compared to ABO videos.
            # thred_std = list_thred_std[ind_video] # SNR threshold used to determine when neurons are active
            (rows, cols) = Dimens[ind_video] # size of the network input and output
            (Lx, Ly) = (rows, cols) # size of the original video

            # %% set pre-processing parameters
            nn = nframes
            gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
            num_median_approx = 900 # number of frames used to caluclate median and median-based standard deviation
            list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.
            filename_TF_template = 'GCaMP6s_spike_tempolate_mean.h5'

            if useTF:
                h5f = h5py.File(filename_TF_template,'r')
                # Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
                # Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
                fs_template = 3
                Poisson_template = np.array(h5f['filter_tempolate']).squeeze()
                h5f.close()
                peak = Poisson_template.argmax()
                length = Poisson_template.shape
                xp = np.arange(-peak,length-peak,1)/fs_template
                x = np.arange(np.round(-peak*rate_hz/fs_template), np.round(length-peak*rate_hz/fs_template), 1)/rate_hz
                Poisson_filt = np.interp(x,xp,Poisson_template)
                Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)].astype('float32')
            else:
                Poisson_filt=np.array([1])
            # dictionary of pre-processing parameters
            Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
                'nn':nn, 'Poisson_filt': Poisson_filt}
            num_total = nframes - len(Poisson_filt) + 1 # number of frames of the video

            # %% set the range of post-processing hyper-parameters to be optimized in
            # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
            list_minArea = list(range(60,175,10)) 
            # average area of a typical neuron (unit: pixels in ABO videos)
            list_avgArea = [177] 
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
            list_thresh_pmap = list(range(130,245,10))
            # threshold to binarize the neuron masks. For each mask, 
            # values higher than "thresh_mask" times the maximum value of the mask are set to one.
            thresh_mask = 0.5
            # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
            thresh_COM0 = 2
            # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
            list_thresh_COM = list(np.arange(4, 9, 1)) 
            # minimum IoU of two masks to be considered the same neuron
            list_thresh_IOU = [0.5] 
            # minimum consecutive number of frames of active neurons
            list_cons = list(range(1, 8, 1)) 

            # adjust the units of the hyper-parameters to pixels in the test videos according to relative magnification
            list_minArea= list(np.round(np.array(list_minArea) * Mag**2))
            list_avgArea= list(np.round(np.array(list_avgArea) * Mag**2))
            thresh_COM0= thresh_COM0 * Mag
            list_thresh_COM= list(np.array(list_thresh_COM) * Mag)
            # adjust the minimum consecutive number of frames according to different frames rates between ABO videos and the test videos
            # list_cons=list(np.round(np.array(list_cons) * rate_hz/30).astype('int'))

            # dictionary of all fixed and searched post-processing parameters.
            Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
                    'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
                    'thresh_mask': thresh_mask, 'list_cons': list_cons}
            print(Params_set)


            # # pre-processing for training
            # # Exp_ID = list_Exp_ID[ind_video]
            # # %% Pre-process video
            # video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
            #     useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #

            # # %% Determine active neurons in all frames using FISSA
            # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
            # generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
            # del video_input

            # %% CNN training
            # if cross_validation == "use_all":
            #     list_CV = [nvideo]
            # else: 
            #     list_CV = list(range(0,nvideo))
            # for CV in list_CV:
            #     if cross_validation == "leave_one_out":
            #         list_Exp_ID_train = list_Exp_ID.copy()
            #         list_Exp_ID_val = [list_Exp_ID_train.pop(CV)]
            #     elif cross_validation == "train_1_test_rest":
            #         list_Exp_ID_val = list_Exp_ID.copy()
            #         list_Exp_ID_train = [list_Exp_ID_val.pop(CV)]
            #     elif cross_validation == "use_all":
            #         list_Exp_ID_val = None
            #         list_Exp_ID_train = list_Exp_ID.copy() 
            #     else:
            #         raise('wrong "cross_validation"')
            #     if not use_validation:
            list_Exp_ID_train = [Exp_ID]
            list_Exp_ID_val = None # Afternatively, we can get rid of validation steps
            file_CNN = weights_path+'Model_{}.h5'.format(Exp_ID)
            file_CNN_2 = weights_path+'Model_CV0.h5'
            results = train_CNN(dir_network_input, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
                BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rows, cols), Params_loss, exist_model)
            copyfile(file_CNN, file_CNN_2)

            # save training and validation loss after each eopch
            f = h5py.File(training_output_path+"training_output_{}.h5".format(Exp_ID), "w")
            f.create_dataset("loss", data=results.history['loss'])
            f.create_dataset("dice_loss", data=results.history['dice_loss'])
            if use_validation:
                f.create_dataset("val_loss", data=results.history['val_loss'])
                f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
            f.close()

            # %% parameter optimization
            parameter_optimization_cross_validation(cross_validation, list_Exp_ID_train, Params_set, \
                (Lx, Ly), (rows, cols), dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
                batch_size_eval, useWT=useWT, useMP=True, load_exist=load_exist)
            # rename 'Optimization_Info_{}.mat'
            # Info_dict = loadmat(dir_output+'Optimization_Info_{}.mat'.format(0))
            # savemat(dir_output+'Optimization_Info_{}.mat'.format(Exp_ID), Info_dict)
            copyfile(dir_output+'Optimization_Info_{}.mat'.format(0), dir_output+'Optimization_Info_{}.mat'.format(Exp_ID))

