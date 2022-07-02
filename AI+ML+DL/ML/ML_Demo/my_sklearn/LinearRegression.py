import numpy as np


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def __repr__(self):
        return "LinearRegression()"

    def fit_