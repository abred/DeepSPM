import os
import png
import sys

import numpy as np

from scipy import ndimage
from scipy import fftpack
from scipy import misc


def max_empty_size(mat, ZERO=0):
    """Find the largest square of ZERO's in the matrix `mat`.
    Source: adapted from: Joy Dutta: https://stackoverflow.com/a/1726667
    """
    # nrows, ncols = len(mat), (len(mat[0]) if mat else 0)
    nrows, ncols = mat.shape
    if not (nrows and ncols):
        return 0  # empty matrix or rows
    # counts = [[0]*ncols for _ in xrange(nrows)]
    counts = np.zeros((nrows, ncols))
    for i in reversed(range(nrows)):     # for each row
        assert len(mat[i]) == ncols  # matrix must be rectangular
        for j in reversed(range(ncols)):  # for each element in the row
            if mat[i][j] == ZERO:
                counts[i][j] = (1 + min(
                    counts[i][j+1],   # east
                    counts[i+1][j],   # south
                    counts[i+1][j+1]  # south-east
                    )) if i < (nrows - 1) and j < (ncols - 1) else 1  # edges
    mx = -1
    lx = -1
    ly = -1
    for row in range(len(counts)):
        for col in range(len(counts[row])):
            if counts[row, col] > mx:
                mx = counts[row, col]
                ly = row
                lx = col

    return mx, ly, lx


# find the largest empty spot in imgT
# compute local variance image
# threshold variance image
# is there an area with a local variance below threshold?
def findEmptySpot(imgT, ep, step, outDir):
    img = np.copy(imgT)

    imgMean = img.mean()
    var = np.sum(np.square(img-imgMean))

    szImg = img.shape[0]

    sz = int(img.shape[0]//4)
    yr = int(img.shape[0]//sz)
    xr = int(img.shape[1]//sz)
    varSampled = np.zeros((yr, xr))
    imgNorm = (img)/(img.max()-img.min())
    varSampled = np.zeros((img.shape[0]-sz, img.shape[1]-sz))

    print("find empty spot in", sz, varSampled.shape)
    for y in range(sz//2, img.shape[0]-sz//2):
        for x in range(sz//2, img.shape[1]-sz//2):
            subImg = imgNorm[y-sz//2:y+sz//2, x-sz//2:x+sz//2]
            subImgMean = subImg.mean()
            subVar = np.sum(np.square(subImg-subImgMean))
            varSampled[y-sz//2, x-sz//2] = subVar

    varSampled = (varSampled-varSampled.min())/(varSampled.max()-varSampled.min())
    varSampled = varSampled * 255.0
    std = np.std(varSampled)

    mn = varSampled.mean()
    mn = varSampled.min()
    varSampled[varSampled <= 10] = 0  # All low values set to 0
    varSampled[varSampled != 0] = 255

    varSampled2 = np.zeros((img.shape[0], img.shape[1]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            varSampled2[y, x] = 128

    varSampled2[sz//2:img.shape[0]-sz//2, sz//2:img.shape[1]-sz//2] = varSampled
    mx, ly, lx = max_empty_size(varSampled2, 0)
    print("empty spot raw", mx, ly, lx)
    lyF = ly
    lxF = lx
    ly += mx//2 - sz//2
    lx += mx//2 - sz//2
    print("empty spot", mx, ly, lx)

    mx = sz
    img = (img-img.min())/(img.max()-img.min())
    img *= 255
    ly = int(ly)
    lx = int(lx)
    for i in range(sz):
        img[ly+i,lx] = 255
        img[ly+i,lx+sz] = 255
        img[ly,lx+i] = 255
        img[ly+sz,lx+i] = 255

    fn = os.path.join(outDir, "imgs",
                      "Ep_" + str(ep) + "-Step_" + str(step) + "-marked.png")
    print("storing file: {}".format(fn))
    png.from_array(img.astype(np.uint8), 'L;8').save(fn)

    fn = os.path.join(outDir, "imgs",
                      "Ep_" + str(ep) + "-Step_" + str(step) + "-emptySpots.png")
    print("storing file: {}".format(fn))
    png.from_array(varSampled2.astype(np.bool_), 'L;8').save(fn)

    return ly, lx, mx


def findEmptySpotAbs(imgT, threshold, imageNameString):
    """look for points that lie on surface, find largest such area"""
    img = np.copy(imgT)

    imgMean = img.mean()
    var = np.sum(np.square(img-imgMean))

    szImg = img.shape[0]

    sz = int(img.shape[0]//4)
    yr = int(img.shape[0]//sz)
    xr = int(img.shape[1]//sz)
    varSampled = np.zeros((yr, xr))
    print('img.max(), img.min()',img.max(), img.min())
    imgNorm = (img)/(img.max()-img.min())
    varSampled = np.zeros((img.shape[0]-sz, img.shape[1]-sz))

    freeImg = np.zeros(shape=img.shape)
    freeImg[:]=128.0

    print("find empty spot in", sz, varSampled.shape)
    for y in range(1,img.shape[0]-1):
        for x in range(1, img.shape[1]-1):
            # We check if points lie on surface
            if(abs(img[y,x]+1.0)>threshold):
                freeImg[y,x]=255.0
            else:
                freeImg[y,x]=0.0


    print("minFS:"+str(img.min())+ " maxFS:"+ str(img.max())+ " range:"+str(img.max() - img.min()) )

    mx, ly, lx = max_empty_size(freeImg, 0)

    print("empty spot raw", mx, ly, lx)
    ly = ly+mx/2
    lx = lx+mx/2
    print("empty spot", mx, ly, lx)

    img = (img-img.min())/(img.max()-img.min())
    img *= 255
    img[int(ly),int(lx)]=255
    img[int(ly-1),int(lx-1)]=255
    img[int(ly-1),int(lx+1)]=255
    img[int(ly+1),int(lx-1)]=255
    img[int(ly+1),int(lx+1)]=255
    for i in range(int(mx)):
        img[int(ly+i-mx/2),int(lx-mx/2)] = 255
        img[int(ly+i-mx/2),int(lx+mx-mx/2)] = 255
        img[int(ly-mx/2),int(lx+i-mx/2)] = 255
        img[int(ly+mx-mx/2),int(lx+i-mx/2)] = 255

    fn = imageNameString + "-marked.png"
    print("storing file: {}".format(fn))
    png.from_array(img.astype(np.uint8), 'L;8').save(fn)

    fn = imageNameString + "-emptySpots.png"
    print("storing file: {}".format(fn))
    png.from_array(freeImg.astype(np.uint8), 'L;8').save(fn)

    return float(ly), float(lx), float(mx)



def findEmptySpotDOG(img, imageNameString):
    """use difference of gaussians to find empty spot"""
    freeImg = spf.gaussian_filter(img, 3.0) - spf.gaussian_filter(img, 3.01)
    freeImg[freeImg <= 0] = 0
    freeImg[freeImg > 0] = 255
    mx, ly, lx = max_empty_size(freeImg, 0)

    print("empty spot raw", mx, ly, lx)

    ly = ly+mx/2
    lx = lx+mx/2
    # print(mx, ly, lx)

    print("empty spot", mx, ly, lx)

    img = (img-img.min())/(img.max()-img.min())
    img *= 255
    ly = int(ly)
    lx = int(lx)
    sz = img.shape[0]
    for i in range(int(mx)):
        img[min(sz-1, int(ly+i-mx/2)),  int(lx-mx/2)] = 255
        img[min(sz-1, int(ly+i-mx/2)),  int(lx+mx-mx/2)] = 255
        img[int(ly-mx/2),    min(sz-1, int(lx+i-mx/2))] = 255
        img[int(ly+mx-mx/2), min(sz-1, int(lx+i-mx/2))] = 255

    fn = imageNameString+ "-marked.png"
    print("storing file: {}".format(fn))
    png.from_array(img.astype(np.uint8), 'L;8').save(fn)

    fn = imageNameString + "-emptySpots.png"
    print("storing file: {}".format(fn))
    png.from_array(freeImg.astype(np.uint8), 'L;8').save(fn)

    return float(ly), float(lx), float(mx)
