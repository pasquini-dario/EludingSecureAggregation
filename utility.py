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

class lr_schlr(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, steps):
        self.learning_rate = initial_learning_rate
        self.steps = steps 

    def __call__(self, step):
        if step in self.steps:
            self.learning_rate = self.learning_rate *.1
            print(f"\t[Scaling learning rate: {np.round(self.learning_rate, 4)}]")
        return self.learning_rate