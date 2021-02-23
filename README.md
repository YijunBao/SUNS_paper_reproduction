![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# SUNS
Shallow UNet Neuron Segmentation (SUNS) is an automatic algorithm to segment active neurons from two-photon calcium imaging videos. It used temporal filtering and whitening schemes to extract temporal features associated with active neurons, and used a compact shallow U-Net to extract spatial features of neurons.

Copyright (C) 2020 Duke University NeuroToolbox

This repo is an archived version of SUNS (v1.0). It also contains the results for our paper and the code to generate those results. This repo will not be updated. Please visit the [newer version](https://github.com/YijunBao/Shallow-UNet-Neuron-Segmentation_SUNS) if you are not interested in reproducing our paper results. 

![Example video](readme/Masks%202%20raw.gif)

- [SUNS](#suns)
- [System requirement](#system-requirement)
- [Installation on Windows or Linux](#installation-on-windows-or-linux)
- [Demo](#demo)
- [Input, Output, and Intermediate Files](#input-output-and-intermediate-files)
  - [Input files](#input-files)
  - [Output files](#output-files)
  - [Intermediate files](#intermediate-files)
- [Use your own data](#use-your-own-data)
  - [Use your own videos](#use-your-own-videos)
  - [Use your own GT masks](#use-your-own-gt-masks)
  - [Use your own temporal filter kernel](#use-your-own-temporal-filter-kernel)
  - [Set the path and video parameters](#set-the-path-and-video-parameters)
  - [Set your own post-processing parameters](#set-your-own-post-processing-parameters)
  - [Set your processing options](#set-your-processing-options)
  - [No training data?](#no-training-data)
  - [Use optional spatial filtering?](#use-optional-spatial-filtering)
- [Links to Datasets and Manual Markings](#links-to-datasets-and-manual-markings)
- [Citing](#citing)
- [Licensing and Copyright](#licensing-and-copyright)
- [Sponsors](#sponsors)


# System requirement
* Memory: ~6x file size of the raw video if the the raw video is in uint16 format. ~3x file size of the raw video if the the raw video is in float32 format. 
* A CUDA compatible GPU is preferred.


# Installation on Windows or Linux
* Install [Anaconda](https://www.anaconda.com/)
* Launch Anaconda prompt and type the following in order (SUNS_python_root_path is the directory to which the provided files were downloaded to, such as `C:/Users/{username}/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS`):
```bat
cd SUNS_python_root_path
cd installation
conda env create -f environment_suns.yml -n suns
```
* Go to the Anaconda environment foler, (e.g., `C:/ProgramData/Anaconda3/envs` or `C:/Users/{username}/.conda/envs`), and then go to folder `suns/Lib/site-packages/fissa`, overwrite `core.py` with the files provided in the `installation` folder. The modified files increase speed by eliminating redundant `separate` or `separation_prep` during initializating an `Experiment` object, and enable videos whose size are larger than 4 GB after converting to float32. If neither of them is important to you, then you can skip replacing the files. If you see a lot of text output when activating suns environment and do not want to see them, you can go to the Anaconda environment foler, go to folder `suns/etc/conda/activate.d`, and delete the two files under this folder. 

The installation should take less than half an hour in total. The first run of the software may take some additional time (up to 20 minutes on a laptop) to add the GPU, but this extra time will not occur in later runs.


# Demo
We provided a demo for all users to get familiar with our software, in the directory `{SUNS_python_root_path}/demo`. We provided four two-photon imaging videos as well as their manually marked neurons in `demo/data`, adapted from [CaImAn dataset](https://zenodo.org/record/1659149). The demo will perform a cross validation over the four videos: train the CNN and search for optimal hyper-parameters using three videos, and test SUNS with the training output on the remaining video. 

To run the demo on Windows, launch Anaconda prompt and type the following script 
```bat
cd SUNS_python_root_path
cd demo
conda activate suns
.\demo_pipeline.bat
```

The demo contains four parts: training CNN and hyper-parameters, testing SUNS batch, testing SUNS online, and testing SUNS online with tracking. The input, output, and intermediate files will be explained in [Input, Output, and Intermediate Files](#input-output-and-intermediate-files). Alternatively, you can also run `demo_pipeline_1to3.bat` instead of `demo_pipeline.bat`. The pipeline `demo_pipeline.bat` does standard leave-one-out cross validation on the four example videos. The pipeline `demo_pipeline_1to3.bat` trains the CNN model and post-processing parameters on one video, and tests on the remaining videos. The expected average F1 scores and example processing time on Windows are shown on the following tables.

Expected average F1 score
|	|Train	|Batch	|Online	|Track|
|:------:|:------:|:------:|:------:|:------:|
|train 3 test 1	|0.77	|0.75	|0.67	|0.65|
|train 1 test 3	|0.81	|0.72	|0.66	|0.66|

Example running time
|CPU	|GPU	|Train 3 to 1<br>(total)	|Train 1 to 3<br>(total)	|Batch<br>(average)	|Online<br>average)	|Track<br>(average)|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|AMD 1920X<br>12-core	|NVIDIA Titan RTX|1.0 h	|0.4 h	|1.5 s	|18 s	|20s	|
|Intel i7-6800K<br>6-core	|NVIDIA GTX 1080|1.5 h	|0.6 h	|1.7 s	|16 s	|20 s	|
|Intel i5-6200U<br>dual-core	|NVIDIA 940MX	|5.1 h	|1.8 h	|4.3 s	|24 s	|28 s|


To run the demo on Linux, launch Anaconda prompt and type the following script 
```sh
cd SUNS_python_root_path
cd demo
conda activate suns
sh demo_pipeline.sh
```
If this does not run through due to multiprocessing errors, another option is to copy the files in the folder `demo/split_version` to the `demo` folder and run `demo_pipeline_split.sh` instead of `demo_pipeline.sh`.

Due to the randomness in training CNN, there is a low chance (~1%) that the training failed (e.g., the resulting F1 score is 0). If that happens, simply retrain the CNN one more time, and it should work out.


# Input, Output, and Intermediate Files
When you download this repo, you will see some files and a folder under `demo/data`. All the input, output, and intermediate files are saved under this folder. 

## Input files
* The .h5 files are the input videos contained in dataset 'mov'. This is a 3D dataset with shape = (T, Lx, Ly), where T is the number of frames, and (Lx, Ly) is the lateral dimension of each frame. The demo videos have (T, Lx, Ly) = (3000, 120, 88). 
* The `GT Masks` folder contains the ground truth (GT) masks of each video. Each set of GT masks is stored in both a 3D array and a 2D sparse matrix. `FinalMasks_YST_part??.mat` stores the GT masks in a 3D array with shape = (Ly, Lx, n_GT) in MATLAB, where Ly and Lx should match the lateral dimensions of the video, and n_GT is the number of GT masks. The demo masks have (Ly, Lx, n_GT) = (88, 120, n_GT). `FinalMasks_YST_part??_sparse.mat` stores the GT masks in a 2D sparse matrix with shape = (Ly\*Lx, n_GT) in MATLAB. The demo GT masks have (Ly\*Lx, n_GT) = (88\*120, n_GT).
* Important notice: The default dimension order for multi-dimensional array is reversed in MATLAB and python. When you save a dataset with shape = (L1, L2, L3) in MATLAB to an .h5 file or a .mat file with version 7.3 or newer (requiring h5py.File to load in python workspace), and then load it in python, the shape will become (L3, L2, L1). However, if you save the dataset as a .mat file with version 7 or earlier (requiring scipy.loadmat to load in python workspace), the dimensions will preserve and still be (L1, L2, L3). In this document, we will use the python default order to describe the datasets in python workspace or saved in .h5, and use the MATLAB default order to describe the datasets saved in .mat. Sometimes you need to transpose the dimensions to make them consistent. In python, you can transpose the dimensions using `Masks = Masks.transpose((2,1,0))`. In MATLAB, you can transpose the dimensions using `Masks = permute(Masks,[3,2,1])`.

## Output files
After running a complete pipeline, the intermediate and output files will be under a new folder `demo/data/noSF`. 
* `output_masks` stores the output masks of SUNS batch. 
* `output_masks online` stores the output masks of SUNS online. 
* `output_masks track` stores the output masks of SUNS online with tracking. 
* The segmented masks are stored in `Output_Masks_{}.mat` as a 3D array with shape (n_o, Lx, Ly) in MATLAB, where Lx and Ly should match the lateral dimensions of the video, and n_o is the number of output masks. The demo output masks should have (n_o, Lx, Ly) = (n_o, 120, 88). The output of SUNS online and SUNS online with tracking also have `list_time_per` in these files, which stores the processing time per frame in the online processing. The processing time per frame is meaningless for the initialization part, so it is set to zero in the initialization part. In the demo, the initilization part contains the first 289 frames. 
* The speed and accuracy scores are summarized in the same three folders as `Output_Info_All.mat`. In this file, `list_Recall`, `list_Precision`, and `list_F1` stores the recall, precision, and F1 scores for the four videos. `list_time` and `list_time_frame` are the total processing time and the average processing time per frame for the four videos. `list_time_frame` is simply `list_time` divided by the number of frames of this video. For the output of SUNS batch, `list_time` and `list_time_frame` are 4x4 arrays, and each row is the processing time of pre-processing, CNN inference, post-processing, and total time of a video. For the output of SUNS online and SUNS online with tracking, `list_time` and `list_time_frame` are 4x3 arrays, and each row is the processing time of initilization stage, frame-by-frame stage, and total time of a video. 

## Intermediate files
After running a complete pipeline, the intermediate and output files will be under a new folder `demo/data/noSF`. Most of intermediate files are numbered according to the video name and/or the index of cross validation. 
* `network_input` stores the SNR videos after pre-processing (in dataset 'network_input') with shape = (T_SNR, Lx, Ly). Here, T_SNR is slightly smaller than T due to temporal filtering. 
* `FISSA` stores the temporary output of [FISSA](https://github.com/rochefort-lab/fissa).
* `traces` stores the traces of the GT neurons after decontamination using FISSA.
* `temporal_masks(3)` stores the temporal masks of active neurons (in dataset 'temporal_masks')used to train CNN.
* `Weights` stores the trained CNN model.
* `training output` stores the loss after each training epoch.
* `temp` stores the recall, precision, and F1 score of all parameter combinations for each video and each cross validation.
* `output_masks` stores the optimal hyper-parameters in `Optimization_Info_{}.mat`. In this file, `Params_set` stores the searching ranges of the hyper-parameters, and `Params` stores the optimized hyper-parameters. [Set your own post-processing parameters](#set-your-own-post-processing-parameters) will describe the parameters in detail. 

# Use your own data
Of course, you can modify the demo scripts to process other videos. You need to set the folders of the videos and GT masks, and change some parameters in the python scripts to correspond to your videos. 

## Use your own videos
* Set the folder and file names of your video correctly. You need to change the variables `dir_video` and `list_Exp_ID` in all the python scripts. The variable `dir_video` is the folder containing the input videos. For example, if your videos are stored in `C:/Users/{username}/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS/demo/data`, set `dir_video = 'C:/Users/{username}/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS/demo/data'`. You can also use relative path, such as `dir_video = './data'`. The variable `list_Exp_ID` is the list of the file names (without extension) of the input videos (e.g., `list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']` in the demo referes to the four input files in `{SUNS_python_root_path}/demo/data/YST_part??.h5`). 
* Currently, we only support .h5 files as the input video, so you need to convert the format of your data to .h5, and save the video in dataset 'mov'. The video should have a shape of (T, Lx, Ly), where T is the number of frames, and (Lx, Ly) is the lateral dimension of each frame. The support to more video formats will come soon. When doing format conversion, make sure the dimension is in the correct order. For example, if you save save the .h5 files from MATLAB, the shape of the dataset should be (Ly, Lx, T) in MATLAB. See [Input files](#input-files) for more explanation of the dimension order.

## Use your own GT masks
* If you want to train your own network, create a folder `GT Masks` under the video folder, and copy all the ground truth markings into that folder. 
* Currently, we only support .mat files as the GT masks, so you need to convert the format of your GT masks to .mat, and save the GT masks in dataset 'FinalMasks'. The name of the masks file should be `FinalMasks_{Exp_ID}.mat`, where the `{Exp_ID}` is the name of the corresponding video. The GT masks should be saved as a 3D array named `FinalMasks`, and the dimension should be (Ly, Lx, n_GT) in MATLAB, where Ly and Lx are the same as the lateral dimension of the video, and n_GT is the number of GT masks.
* The masks created by some manual labeling software may contain an empty (all zero) image as the first frame. You need to remove the empty frame before saving them.
* Because we mainly use sparse matrix representations of the masks to speed up evaluation, you also need to convert the masks from a 3D array to a 2D sparse matrix. You can modify the MATLAB script `utils/generate_sparse_GT.m` to do that by setting `dir_Masks` to the folder `GT Masks` (e.g., `dir_Masks = '..\demo\data\GT Masks\'` in the demo). The sparse matrices are saved as .mat files in 'v7' format. Please do not save them in 'v7.3' format, otherwise, you may need to use h5sparese (not installed) to read the files in python.

## Use your own temporal filter kernel
* We used a matched filter as our temporal filtering scheme, so we need a filter template, which is the reversed version of the temporal filter kernal. In the demo, we used `demo/YST_spike_tempolate.h5` to store the matched filter template, and you can use this as the starting point. We determined the filter template by averaging calcium transients within a moderate SNR range from the videos. The filter template was generated by the MATLAB scripts `utils/temporal filter/calculate_traces_bgtraces_demo.m` and `utils/temporal filter/temporal_filter_demo.m`. You can modify these scripts to generate the filter template for your dataset. In both scripts, you need to specify the folder of the video (e.g., `dir_video='..\..\demo\data\'`), the folder of the GT Masks (should be `dir_GTMasks=fullfile(dir_video,'GT Masks\')` according to our previous instruction), and the list of video names (e.g., `list_Exp_ID={'YST_part11';'YST_part12';'YST_part21';'YST_part22'}`). In `utils/temporal filter/temporal_filter_demo.m`, you also need to specify the frame rate of the video (e.g., `fs=10`), number of frames before spike peak (e.g., `before=15`), number of frames after spike peak (e.g., `after=60`), the minimum and maximum allowed SNR (e.g., `list_d=[5,6]`), and the file name of the saved filter template (e.g., `h5_name = 'YST_spike_tempolate.h5'`). You also need to change the variable `filename_TF_template` in all the python scripts to your saved filter template file (e.g. `filename_TF_template = './YST_spike_tempolate.h5'`). 
* Alternatively, you can also use a simple exponential or double exponential curve as the filter template. We provided an example exponentially decay template in comments of the demo scripts. You need to find the line `# Alternative temporal filter kernel using a single exponential decay function`, comment the five lines before this line, and uncomment the four lines after this line. You need to set appropriate decay time constant (e.g. `decay = 0.8` for GCaMP6s)). Make sure the make the same changes for all the training and testing scripts. 
* If the quality of your video is very good, and the signal to noise ratio is high, it is OK to not use temporal filtering. If you decide not to use temporal filtering, you can set `useTF=False` in all python scripts. 

## Set the path and video parameters
In all the python scripts, you need to specify a few path and video parameters:
* The list of video names (e.g., `list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']`)
* The folder of the video (e.g., `dir_video='.\data\'`);
* The folder of the GT Masks (should be `dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_')` according to our previous instruction);
* The frame rate of the video (e.g., `rate_hz=10`);
* Spatial magnification compared to ABO videos (e.g., `Mag=6/8`). If the pixel size of your video is `pixel_size` um/pixel, then set `Mag = 0.785 / pixel_size`.

## Set your own post-processing parameters
SUNS is a supervised learning algorithm, but the post-processing parameters highly affect the accuracy of the final output masks. These parameters include:
* `minArea`: minimum area of a neuron
* `avgArea`: average area of a typical neuron
* `thresh_pmap`: uint8 threshould of probablity map
* `thresh_COM`: maximum COM distance of two masks to be considered the same neuron
* `cons`: minimum consecutive number of frames of active neurons

We determined `avgArea` according to the typical neuron area reported in the literature, and optimized other parameters using grid search by maximizing the average F1 score on the training videos. You can change the search ranges of these parameters by changing the variables `list_{parameter}`, where `{paraemter}` is one of the above parameters. The demo uses `list_avgArea = [177] `, `list_minArea = list(range(30,85,5)) `, `list_thresh_pmap = list(range(130,235,10))`, `list_thresh_COM = list(np.arange(4, 9, 1)) `, and `list_cons = list(range(1, 8, 1))`.

## Set your processing options
* `useSF` indicates whether spatial filtering is used in pre-processing (False in demo);
* `useTF` indicates whether temporal filtering is used in pre-processing (True in demo);
* `BATCH_SIZE` is the batch size in CNN training; `batch_size_eval` is the batch size in CNN inference in SUNS batch; `batch_size_init` is the batch size in CNN inference in the initialization stage of SUNS online and SUNS online with tracking. You can adjust these values according to the GPU memory capacity of your computer; 
* `thred_std` is SNR threshold used to determine when neurons are active to create temporal masks for training (3 in demo). When the video is longer or have higher signal to noise quality, you can increase this value;
* `cross_validation` determines the cross-validation strategy ("leave-one-out" means training on all but one video and testing on that one video, "train_1_test_rest" means training on one video and testing on the other videos, and "use_all" means training on all videos and testing on other videos not in the list);
* `use_validation` indicates if a validation set outside the training set is used in training CNN for validation purpose (True in demo); it does not affect the training result;
* `update_baseline` indicates whether the baseline and noise are updated every a few frames during the online processing (False in demo).

You can read the code for explanations of other options and parameters.

## No training data?
* SUNS requires a CNN model and post-processing parameters to process a video. It is best to train your own CNN models and post-processing parameters for your dataset, because the optimal CNN models and post-processing parameters, especially parameters, can be different with different imaging conditions. Therefore, our expectation is that you manually label at least one video recorded by your two-photon microscope, and use that to train the CNN and parameters. 
* However, at the beginning, you can also use the CNN model and the parameters we provided in the folder `training results`. You can do so by setting the path of the CNN model as `filename_CNN = 'training results/Weights/Model_CV10.h5'` and the path of the optimal post-processing parameters as `Optimization_Info = 'training results/output_masks/Optimization_Info_10.mat'` in the test demo script. We don't recommnd using the training results from the demo videos, because the demo videos are relatively small. Our recommended CNN model and hyper-parameters were trained from 10 large videos, so they should perform better in general. 

## Use optional spatial filtering?
We have an optional spatial filtering step in the pre-processing. Althought it did not improve the performance in many of our experiements, it sometimes can improve the accuracy. If you want to try spatial filtering, you need to set `useSF=True` in all python scripts. We recommend you to uncommend the first block of `demo_pipeline.bat` or `demo_pipeline.sh`, so that you can run `demo_learn_wisdom.py` and `demo_learn_wisdom_2d.py`. Make sure to change the dimensions in these scripts to fit your videos before running (e.g., `Dimens = (120,88)` and `Nframes = 3000`). After running them, you will find some text files under `demo/wisdom`, which stores the learned wisdom used for spatial filtering. When you have these files with correct dimensions of your video, the preparation of spatial filtering will be faster.


# Links to Datasets and Manual Markings
In our paper, we used two-photon imaging videos from [Allen Brain Observatory dataset](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS), [Neurofinder Challenge website](https://github.com/codeneuro/neurofinder), and [CaImAn dataset](https://zenodo.org/record/1659149). We used the manual markings of Allen Brain Observatory and Neurofinder from [STNeuroNet](https://github.com/soltanianzadeh/STNeuroNet) repository, and used the manual markings of CaImAn dataset from [CaImAn dataset](https://zenodo.org/record/1659149). A more detailed instruction is given under the folder `paper reproduction`. 


# Citing 
If you use any part of this software in your work, please cite Bao et al. 2020:


# Licensing and Copyright
SUNS is released under [the GNU License, Version 2.0](LICENSE).


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
