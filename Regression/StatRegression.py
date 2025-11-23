from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple
from numpy.typing import NDArray

def lin_regression(data: Tuple[NDArray, NDArray]):
    lin_reg = LinearRegression().fit(*data)