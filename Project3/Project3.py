'''from idlelib.runscript import indent_message

import cv2 as cv
from cv2.ximgproc import anisotropicDiffusion
import numpy as np

img = cv.imread('46_colon_40x.tif')
filtered = anisotropicDiffusion(img,
    alpha=0.1,
    K=50,
    niters=10
)
gray = cv.cvtColor(filtered,cv.COLOR_BGR2GRAY)
th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,61,0.8)

def Clean(opened,condition):
    # Source - https://stackoverflow.com/a/42812226
    # Posted by Soltius, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-06, License - CC BY-SA 4.0
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(opened)

    sizes = stats[:, cv.CC_STAT_AREA]

    im_result = np.zeros_like(im_with_separated_blobs).astype(np.uint8)

    for index_blob in range(1, nb_blobs):
        if condition(sizes[index_blob]):
            im_result[im_with_separated_blobs == index_blob] = 255
    return im_result

im_result = Clean(th3,lambda x: x>2500)
negated = 255 - im_result
finitto = Clean(negated,lambda x: x<5000)


# Kopia oryginalnego obrazu
result = img.copy()
# Krawędzie zaznaczone na czerwono
edges = cv.Canny(finitto, 100, 200)


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
opened = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
bigCircles = Clean(opened,lambda x: x > 4000)

edges1 = cv.Canny(bigCircles, 100, 200)

# Krawędzie zaznaczone na czerwono
result[edges1 > 0] = [0, 255, 0]

result[edges > 0] = [0, 0, 255]



cv.imshow("Original", img)
cv.imshow("Edges", edges)
cv.imshow("Overlay", result)

#cv.imshow('image', img) #cool feature
#cv.imshow('filt', filtered) #cool feature
#cv.imshow('gray', gray) #cool feature
#cv.imshow('tresh', th3) #cool feature
#cv.imshow('opened', opened)
cv.imshow('deleted', im_result)
#cv.imshow('closed', closed)
#cv.imshow('negated', negated)
#cv.imshow('final', finitto)
cv.waitKey(0)
cv.destroyAllWindows()
'''
###########https://docs.opencv.org/4.13.0/d1/d5c/tutorial_py_kmeans_opencv.html
import cv2 as cv
from cv2.ximgproc import anisotropicDiffusion
import numpy as np

img = cv.imread('46_colon_40x.tif')
filtered = anisotropicDiffusion(img,
    alpha=0.1,
    K=50,
    niters=10
)


colors = filtered.reshape(img.shape[0]*img.shape[1],img.shape[2]).astype(np.float32)
Z = colors
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

# Apply KMeans
ret, label, center = cv.kmeans(Z,3, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()
print(center)
print(np.unique(res2))