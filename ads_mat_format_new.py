""" ads_mat_format_new.py
This extracts the data from the MATLAB output from ADS (S, Y, Z, see ADS.mat for an example)
and saves it as Numpy arrays to be used in Vector fitting (output_s and output_f).

Date: 3/7/2020
Author: Jennifer Houle

"""

import scipy.io
import numpy as np


def load_ads_data(fname) -> tuple:
    mat_formatted = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)

    only_relevant_key = list(mat_formatted.keys())[-1]
    blocks = mat_formatted[only_relevant_key].dataBlocks

    indices = [idx for idx, item in enumerate(blocks.dependents) if item.startswith('Y')]

    s = blocks.data.independent
    y = blocks.data.dependents[indices]
    return s, y


if __name__ == '__main__':
    """ Load data from ADS saved as a MATLAB output by specifying the path/.mat file
    
    Save the files as Numpy arrays
    """

    path = "C:\\Users\\Users"
    a = scipy.io.loadmat(f"{path}\\ADS.mat", struct_as_record=False, squeeze_me=True)

    print(a)
    key = list(a.keys())[-1]
    network = a[key]
    # data_blocks = network['dataBlocks']
    print(network.dataBlocks.dependents)

    s, y = load_ads_data(f"{path}\\ADS.mat")  # fancy new way

    y = network.dataBlocks.data.dependents[1]
    s = network.dataBlocks.data.independent

    np.save('output_s', s)
    np.save('output_f', y)