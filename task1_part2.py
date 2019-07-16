import argparse
import os
import pydicom
from pydicom.data import get_testdata_files
import glob
import numpy as np
import h5py
from utils import *
import sys

parser = argparse.ArgumentParser(description='Saving hdf5 file in the range of another DICOM file')
parser.add_argument('--input-dicom','--i', help='path to input DICOM directory')
parser.add_argument('--input-hdf5','--h', help='path to input hdf5 file')
parser.add_argument('--output-dicom', '--o', help='path to output DICOM directory')
global args
args = parser.parse_args()

# print(args.input_dicom)
# print(args.input_hdf5)
# print(args.output_dicom)

key='data'
hdf5FileName=args.input_hdf5
inputPathTemplate=args.input_dicom
outputPath=args.output_dicom

slicesTemplate,imNamesToWrite=readMag(inputPathTemplate)
volumeTemplate=stack3dArray(slicesTemplate)
volumeArray=readHDF5(hdf5FileName,key)
convertedHDF5=convertRange(volumeArray,volumeTemplate.min(),volumeTemplate.max(),volumeTemplate.dtype)
writeDicom(outputPath,slicesTemplate,convertedHDF5)
