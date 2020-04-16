import numpy as np

from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model, datasets

class RansacWrapper:
    def __init__(self, sz, th=None):
        self.X = np.zeros((sz*sz,2))
        if th is None:
            th = 5e-12
        elif  th < 0:
            th = None
        for i in range(sz):
            for j in range(sz):
                self.X[i*sz+j][0] = i
                self.X[i*sz+j][1] = j
        self.coords = list(np.ndindex(sz,sz))
        self.ransac = Pipeline(
            [('poly', PolynomialFeatures(degree=1)),
             ('ransac', RANSACRegressor(residual_threshold=th,
                                        max_trials=1000))])

    def apply(self, img, params=None):
        sz = img.shape[0]
        y = img.ravel()
        self.ransac.fit(self.X, y)
        plane = self.ransac.predict(self.coords)
        plane.shape = img.shape

        return img-plane
