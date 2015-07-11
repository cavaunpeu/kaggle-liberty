import numpy as np


class BaseTargetTransform(object):

    @classmethod
    def transform(cls, y):
        return y

    @classmethod
    def transform_back(cls, y):
        return y


class SquareRootTargetTransform(BaseTargetTransform):

    @classmethod
    def transform(cls, y):
        y = np.clip(y, 0, np.inf)
        return [np.sqrt(yy) for yy in y]

    @classmethod
    def transform_back(cls, y):
        return [yy**2 for yy in y]


class LogTargetTransform(BaseTargetTransform):

    @classmethod
    def transform(cls, y):
        y = np.clip(y, 0, np.inf)
        return [np.log(yy+1) for yy in y]

    @classmethod
    def transform_back(cls, y):
        return [np.exp(yy)-1 for yy in y]
