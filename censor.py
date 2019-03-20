import numpy as np
import math

"""
Censor labels
    @param truth: dictionary containing label data
    @param censorP: proportion of vertices to censor

    @return: array of keys in truth that were censored
"""


def censor(truth, censorP):
    num_censor = math.floor(censorP * len(truth.keys()))
    censored = np.random.choice(list(truth.keys()), size=num_censor, replace=False)
    return censored
