import argparse
import os
import pydicom
from pydicom.data import get_testdata_files
import glob
import numpy as np
import h5py
from utils import *
import sys

parser = argparse.ArgumentParser(description='Reading DICOM and saving as hdf5 file')
parser.add_argument('--input-dicom','--i', help='path to input DICOM directory')
parser.add_argument('--output-hdf5','--h', help='path to output hdf5 file')
global args
args = parser.parse_args()

# print(args.input_dicom)
# print(args.output_hdf5)

key='data'
hdf5FileName=args.output_hdf5
inputPath=args.input_dicom
slices,imNames=readMag(inputPath)
volume=stack3dArray(slices)
convertedVolume=convertRange(volume,0,1,np.float32)
writeHDF5(hdf5FileName,key,convertedVolume)