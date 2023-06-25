# Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/drug_target_interaction/sign/utils.py
import numpy as np
from sklearn.linear_model import LinearRegression

import torch
from math import sqrt
#import sklearn

def cos_formula(a, b, c):
    ''' formula to calculate the angle between two edges
        a and b are the edge lengths, c is the angle length.
    '''
    res = (a**2 + b**2 - c**2) / (2 * a * b)
    # sanity check
    res = -1. if res < -1. else res
    res = 1. if res > 1. else res
    return np.arccos(res)

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def generate_segment_id(index):    
    zeros = torch.zeros(index[-1].item() + 1, dtype=torch.int)
    index = index[:-1]
    update = torch.ones_like(index,dtype=torch.int)
    for i in range(index.shape[0]):
        zeros[index[i].item()] = + update[i]
    #segments = zeros
    segments =torch.cumsum(zeros,dim=0)[:-1] - 1
    return segments



def get_index_from_counts(counts,device):
    """Return index generated from counts
    This function return the index from given counts.
    For example, when counts = [ 2, 3, 4], return [0, 2, 5, 9]
    Args:
        counts: numpy.ndarray of paddle.Tensor
    Return:
        Return idnex of the counts
    """
    
    if torch.is_tensor(counts):        
        index = torch.cat(
            [
                torch.zeros((1,), dtype=counts.dtype).to(device), 
                torch.cumsum(counts,dim=0).to(device)
            ],
            dim= 0).to(device)
    else:
        index = np.cumsum(counts, dtype="int64")
        index = np.insert(index, 0, 0)
    return index


def segment_pool(x,y):
    assert x.shape[0] == y.shape[0]    
    d  = torch.unique(y).size()[0]    
    c = []
    for i in range(d):
        pt = torch.where(y == i)[0]
        c.append(x[pt,:].sum(dim = 0))
    return torch.stack(c,dim = 0)    


