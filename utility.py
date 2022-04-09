import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def clone_list_tensors(A):
    n = len(A)
    B = [None] * n
    for i in range(n):
        B[i] = tf.identity(A[i])
    return B

def assign_list_tensors(A, B):
    """ A <- B """
    assert len(A) == len(B)
    n = len(A)
    for i in range(n):
        A[i].assign(B[i])
        
def init_list_variables(A):
    n = len(A)
    B = [None] * n
    for i in range(n):
        B[i] = tf.zeros(A[i].shape, dtype=A[i].dtype)
    return B
        
def sum_list_tensors(X):
    agg = init_list_variables(X[0])
    n = len(X)
    m = len(agg)
    
    for i in range(n):
        for j in range(m):
            agg[j] += X[i][j]     
    return agg        

def deepCopyModel(model):
    _model = tf.keras.models.clone_model(model)
    n = len(model.trainable_variables)
    for i in range(n):
        _model.trainable_variables[i].assign(model.trainable_variables[i])        
    return _model

def check_gradient(aggregation, target_gradient, verbose=True):
    assert len(aggregation) == len(target_gradient)
    n = len(aggregation)
    
    num_par = 0
    different_par = 0
    for i in range(n):
        agg_i = aggregation[i].reshape(-1)
        tar_i = target_gradient[i].reshape(-1)
        
        assert len(agg_i) == len(tar_i)
        
        num_par += len(agg_i)
        diff = (agg_i != tar_i).sum()
        different_par += diff
        
        if verbose:
            print(f'layer: {i} with shape {agg_i.shape} recovered?: {diff==0}')
            
    recovered = 1. - (diff / num_par)
    return recovered

