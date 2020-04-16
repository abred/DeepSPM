import numpy as np

def normImg1(img, fac=1e-10, clip=True):
    if clip:
        return np.clip( (img/fac)-1, -1.5, 1.5)
    else:
        return (img/fac)-1
