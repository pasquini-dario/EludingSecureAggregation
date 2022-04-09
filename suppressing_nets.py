import tensorflow as tf
import numpy as np
from utility import *


def zero_all_network(var):
    var = init_list_variables(model.trainable_variables)
    return var

def zero_kernel_i(var, i):
    var[i] = tf.zeros(var[i].shape)
    return var
    