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

path_NIfTI = 'ct.nii'
file_NIfTI = nib.load(path_NIfTI)
img_NIfTI = file_NIfTI.get_fdata()
#print(img_NIfTI)

def Extract_info(img,path,title):
    print("Dimensions: "+str(img.shape))
    print("Type: "+str(type(img.ravel()[0])))
    print("Minimum value: "+str(np.min(img)))
    print("Maximum value: "+str(np.max(img)))
    print("Mean value: "+str(np.mean(img)))
    print("Standard deviation: "+str(np.std(img)))
    print("File size: "+str(os.path.getsize(path)/1024**2))
    plt.hist(img.ravel(), bins=256)
    plt.title("Histogram of "+title)
    plt.xlabel("Pixel value")
    plt.savefig("Histogram_"+title)
    plt.clf()
    plt.hist(img.ravel(), bins=256)
    plt.title("log(Histogram) of "+title) #just to see better, AI was used to check how to set the Y-axis to logarithmic
    plt.xlabel("Pixel value")
    plt.yscale('log')
    plt.savefig("log_Histogram_"+title)
    plt.clf()

#Extract_info(img_TIFF,path_TIFF,"TIFF")
#Extract_info(img_DICOM,path_DICOM,"DICOM")
#Extract_info(img_NIfTI,path_NIfTI,"NIfTI")

def Window(img,lower_threshold,upper_threshold):
    img_normalized = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img_subtracted = img_normalized-lower_threshold*np.ones(img.shape)
    img_multi = img_subtracted*255/(upper_threshold-lower_threshold)
    img_thresholded = np.maximum(0,np.minimum(255,img_multi))
    return img_thresholded.astype(np.uint8)

NIfTI_slice = img_NIfTI[100].copy()

original_imgs = {"NIfTI":NIfTI_slice,
                 "DICOM":img_DICOM,
                 "TIFF":img_TIFF}

processed_imgs={"NIfTI":cv.applyColorMap( Window(NIfTI_slice,85,120),cv.COLORMAP_MAGMA),
                 "DICOM":cv.applyColorMap(Window(img_DICOM,70,100),cv.COLORMAP_PARULA),
                 "TIFF":cv.applyColorMap(Window(img_TIFF,50, 100),cv.COLORMAP_PINK)}

for key,img in original_imgs.items():
    cv.imshow("original "+key,img)
    cv.imwrite("original "+key+".png",img)
    cv.imshow("processed "+key,processed_imgs[key])
    cv.imwrite("processed "+key+".png",processed_imgs[key] )

cv.waitKey(0)
cv.destroyAllWindows()