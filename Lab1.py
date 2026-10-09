import cv2 as cv
import os
import pylibjpeg
import numpy as np
import pydicom as dicom
import nibabel as nib
import matplotlib.pyplot as plt

path_TIFF = 'cell_00077.tif'
img_TIFF = cv.imread(path_TIFF)

path_DICOM = '1-124.dcm'
file_DICOM = dicom.dcmread(path_DICOM)
img_DICOM = file_DICOM.pixel_array
#print(img_DICOM)

path_NIFTI = 'ct.nii'
file_NIFTI = nib.load(path_NIFTI)
img_NIFTI = file_NIFTI.get_fdata()
#print(img_NIFTI)

def Extract_info(img,path):
    print("Dimensions: "+str(img.shape))
    print("Type: "+str(type(img.ravel()[0])))
    print("Minimum value: "+str(np.min(img)))
    print("Maximum value: "+str(np.max(img)))
    print("Mean value: "+str(np.mean(img)))
    print("Standard deviation: "+str(np.std(img)))
    print("File size: "+str(os.path.getsize(path)/1024**2))

#Extract_info(img_TIFF,path_TIFF)
#Extract_info(img_DICOM,path_DICOM)
#Extract_info(img_NIFTI,path_NIFTI)

#plt.hist(img_TIFF.ravel(), bins=256)
#plt.title("Histogram of TIFF")
#plt.xlabel("Pixel value")
#plt.show()

NIFTI_slice = img_NIFTI[100].copy()
cv.imshow('NIFTI_slice',cv.applyColorMap(cv.normalize(
    NIFTI_slice, None, 0, 255, cv.NORM_MINMAX
).astype(np.uint8),cv.COLORMAP_MAGMA))
cv.imshow('DICOM',cv.applyColorMap(cv.normalize(
    img_DICOM, None, 0, 255, cv.NORM_MINMAX
).astype(np.uint8),cv.COLORMAP_PARULA))
cv.imshow('TIFF',cv.applyColorMap(cv.normalize(
    img_TIFF, None, 0, 255, cv.NORM_MINMAX
).astype(np.uint8),cv.COLORMAP_PINK))
cv.waitKey(0)