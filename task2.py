import os
import pydicom
from pydicom.data import get_testdata_files
import glob
import numpy as np
import h5py
from utils import *
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse

def blurring3d(input3d: np.ndarray, sigma: float) -> np.ndarray:
    blurred=np.zeros(input3d.shape,dtype=float)
    for i in range(input3d.shape[0]):
        blurred[i,:,:]=gaussian_filter(input3d[i,:,:],sigma=sigma)
    return blurred


parser = argparse.ArgumentParser(description='Reading DICOM and blurring and saving as DICOM file')
parser.add_argument('--input-dicom','--i', help='path to input DICOM directory')
parser.add_argument('--output-dicom','--o', help='path to output DICOM directory')
global args
args = parser.parse_args()

key='data'
inputPath=args.input_dicom
slices,imNames=readMag(inputPath)
volume=stack3dArray(slices)
convertedVolume=convertRange(volume,0,1,np.float32)
blurredIm=blurring3d(convertedVolume,5)

outputPath=args.output_dicom
convertedBlurred=convertRange(blurredIm,volume.min(),volume.max(),volume.dtype)
writeDicom(outputPath,slices,convertedBlurred)