import os
import cv2
import time
import pyfftw
import multiprocessing as mp


''' This script is used to learn 2D wistom used to speed up FFT-based spatial homomorphic filtering
    2D wistom is only used in SUNS online.
'''
dir_wisdom = 'wisdom\\'
if not os.path.exists(dir_wisdom):
    os.makedirs(dir_wisdom) 
Dimens = [(504,504), (464,504), (416,480)] # lateral dimension of the video

for ind_video in range(0,len(Dimens)): # 
    start = time.time()
    rows, cols = Dimens[ind_video]
    # lateral dimensions slightly larger than the raw video but faster for FFT
    x = cv2.getOptimalDFTSize(rows)
    y = cv2.getOptimalDFTSize(cols)

    # learn 2D wisdom
    start1 = time.time()
    bb = pyfftw.zeros_aligned((x, y), dtype='float32', n=8) # numpy array storing the real-space data
    bf = pyfftw.zeros_aligned((x, y//2+1), dtype='complex64', n=8) # numpy array storing the Fourier-space data
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2,-1), flags=('FFTW_MEASURE',), \
        direction='FFTW_FORWARD',threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2,-1), flags=('FFTW_MEASURE',), \
        direction='FFTW_BACKWARD',threads=mp.cpu_count())
    end1 = time.time()

    # Save the learned 2D wisdom result to "wisdom" folder in three ".txt" files
    bb = pyfftw.export_wisdom()
    print(bb)
    Length_data=str((x, y))
    file = open(dir_wisdom+Length_data+"x1.txt", "wb")
    file.write(bb[0])
    file.close
    file = open(dir_wisdom+Length_data+"x2.txt", "wb")
    file.write(bb[1])
    file.close
    file = open(dir_wisdom+Length_data+"x3.txt", "wb")
    file.write(bb[2])
    file.close()
    print(end1-start1, ' s')
