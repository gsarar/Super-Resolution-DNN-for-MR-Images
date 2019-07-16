import os
import pydicom
from pydicom.data import get_testdata_files
import glob
import numpy as np
import h5py

def readMag(inputPath):
    allSlices=[]
    sliceLocations=[]
    #Reading the files
    imageNames=[]
    for file in glob.glob1(inputPath,"*.mag"):
        imToRead=os.path.join(inputPath, file)
    #     print(imToRead)
        sliceObj=pydicom.dcmread(imToRead)
        allSlices.append(sliceObj)
        sliceLocations.append(sliceObj.SliceLocation)
        imageNames.append(imToRead)
    #The example is sorted, but sorting in any case for other possible no-sorted samples
    #for sorted parameter reverse: A Boolean. False will sort ascending, True will sort descending. Default is False
    sortedSlices=[x for _,x in sorted(zip(sliceLocations,allSlices))]
    return sortedSlices,imageNames

def stack3dArray(sortedSlices):
#     stack to 3d Volume
    volume3d = np.stack([s.pixel_array for s in sortedSlices])
    return volume3d

def convertRange(origVolume,minVal,maxVal,newType):
#     converting to 32-bit float
#     print(origVolume.shape)
    volume3d=origVolume.astype(np.float32) 
#     print(volume3d.shape)
#     normalizing between 0.0 and 1.0
    volume3d-=volume3d.min()
    volume3d/=volume3d.max()
#     normalizing to given range
    volume3d*=(maxVal-minVal)
    volume3d+=minVal
#     print(volume3d.min(),volume3d.max())
#     print(volume3d.shape)
    volume3d=volume3d.astype(newType) 
#     print(volume3d.shape)
    return volume3d

def writeHDF5(filename,key,dataset):
    hf = h5py.File(filename, 'w')
    hf.create_dataset(key, data=dataset)
    hf.close()
    
def readHDF5(filename,key):
    hf = h5py.File(filename, 'r')
    data=np.array(hf[key])
    hf.close()
    return data

def writeDicom(outputPath,slices,convertedHDF5):
    slicesTemplate=slices.copy()
    try:
        # Create target Directory
        os.mkdir(outputPath)
        print("Directory ", outputPath, " Created ") 
    except FileExistsError:
        print("Directory ", outputPath, " already exists")
    for i in range(convertedHDF5.shape[0]):
        slicesTemplate[i].PixelData=convertedHDF5[i,:,:].tostring()
        slicesTemplate[i].save_as(os.path.join(outputPath,'%d.mag'%i))