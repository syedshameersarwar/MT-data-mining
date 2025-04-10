import numpy as np
from scipy.interpolate import interp1d


def index_to_xdata(xdata, indices):
    """
    Interpolate xdata to indices, Useful for converting scipy signal width back from samples to signal time index
    """
    ind = np.arange(len(xdata))
    f = interp1d(ind, xdata)
    return f(indices)
