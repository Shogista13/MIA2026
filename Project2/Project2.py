import cv2 as cv
import pylibjpeg
import numpy as np
import pydicom as dicom
import nibabel as nib

path_TIFF = r'C:\Users\student\MIA2026\Project2\cell_00077.tif'
img_TIFF = cv.imread(path_TIFF)

path_DICOM = r'C:\Users\student\MIA2026\Project2\1-124.dcm'
file_DICOM = dicom.dcmread(path_DICOM)
img_DICOM = file_DICOM.pixel_array

path_NIfTI = r'C:\Users\student\MIA2026\Project2\ct.nii'
file_NIfTI = nib.load(path_NIfTI)
img_NIfTI = file_NIfTI.get_fdata()

def Preprocess(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    normalized_image = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype('uint8')
    return normalized_image

def Threshold(img,thresh='otsu'):
    if thresh=='otsu':
        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        img[img<thresh] = 0
        img[img>thresh] = 255
    return img

def RegionGrowing(img,seed_point,condition,nbr):
    if nbr == 4:
        nhood = ((-1,0),(1,0),(0,-1),(0,1))
    elif nbr == 8:
        nhood = ((-1,1),(1,1),(-1,-1),(1,-1),(-1,0),(1,0),(0,-1),(0,1))
    visited = set(seed_point)
    queue = [seed_point]
    segmented = np.zeros(img.shape)
    while len(queue) > 0:
        point = queue.pop(0)
        if point not in visited:
            for neighbor in nhood:
                neighbor_x = point[0] + neighbor[0]
                neighbor_y = point[1] + neighbor[1]
                if 0<neighbor_x<img.shape[0] and 0<neighbor_y<img.shape[1]:
                    if condition(img,seed_point,(neighbor_x,neighbor_y)):
                        queue.append((point[0]+neighbor[0],point[1]+neighbor[1]))
                        segmented[point[0]+neighbor[0],point[1]+neighbor[1]]=255
                        visited.add(point)
    return segmented

TIFF_prepared = Preprocess(img_TIFF.copy())
thresholded_TIFF = Threshold(TIFF_prepared.copy(),100)
cv.imshow('Thresholded TIFF',thresholded_TIFF)
cv.imwrite('Project2/thresholded_TIFF.png',thresholded_TIFF)
cv.waitKey(0)

DICOM_prepared = Preprocess(img_DICOM)
thresholded_DICOM = Threshold(DICOM_prepared,120)
cv.imshow('Thresholded DICOM',thresholded_DICOM)
cv.imwrite('Project2/thresholded_DICOM.png',thresholded_DICOM)
cv.waitKey(0)

NIfTI_prepared = Preprocess(img_NIfTI[:,100])
rot = cv.getRotationMatrix2D((NIfTI_prepared.shape[0]//2,NIfTI_prepared.shape[1]//2), 90, 1)
thresholded_NIfTI = cv.warpAffine(Threshold(NIfTI_prepared,129),rot,(NIfTI_prepared.shape[0],NIfTI_prepared.shape[1]))
cv.imshow('Thresholded NIfTI',thresholded_NIfTI)
cv.imwrite('Project2/thresholded_NIfTI.png',thresholded_NIfTI)
cv.waitKey(0)
cv.destroyAllWindows()

def MaxDifference(img,pt1,pt2):
    value = abs(int(img[pt1])-int(img[pt2]))<50
    return value

"""def GROW(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        RegionGrowing(img,(x,y),100,MaxDifference,4)"""
#
#cv.imshow('Prepared TIFF',TIFF_prepared)
#cv.setMouseCallback('Prepared TIFF',GROW)
#cv.waitKey(0)
#cv.destroyAllWindows()

grown = RegionGrowing(TIFF_prepared, (100, 100), MaxDifference, 4)
cv.imshow('GROWN',grown)
cv.waitKey(0)
cv.destroyAllWindows()