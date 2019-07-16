import argparse
import os
import pydicom
from pydicom.data import get_testdata_files
import glob
import numpy as np
import h5py
from utils import *
import sys

parser = argparse.ArgumentParser(description='checking if 2 DICOM files are the same')
parser.add_argument('--input-dicom1','--i1', help='path to first input DICOM directory')
parser.add_argument('--input-dicom2','--i2', help='path to second input DICOM directory')
global args
args = parser.parse_args()

# print(args.input_dicom1)
# print(args.input_dicom2)

Path1=args.input_dicom1
slices,imNames=readMag(Path1)
volume=stack3dArray(slices)
#
Path2=args.input_dicom2
slicesTemplate3,imNamesToWrite3=readMag(Path2)
volumeTemplate3=stack3dArray(slicesTemplate3)
convertedVolumeCheck=convertRange(volume,volumeTemplate3.min(),volumeTemplate3.max(),volumeTemplate3.dtype)
print((convertedVolumeCheck==volumeTemplate3).sum()==volume.shape[0]*volume.shape[1]*volume.shape[2])