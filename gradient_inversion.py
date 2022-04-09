import tensorflow as tf
import numpy as np
import tqdm
import math
import itertools
import matplotlib.pyplot as plt

from utility import *


def plot(X, **imshowkargs):
    n = len(X)
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=-.05)
    for i in range(n):
        ax[i].imshow((X[i]), **imshowkargs);  
        ax[i].set(xticks=[], yticks=[])
        ax[i].set_aspect('equal')
        
    return fig, ax


collapse = lambda g : tf.concat([tf.reshape(x, -1) for x in g], 0)

def make_loss(y_, y):
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y, y_)
    loss = tf.reduce_mean(loss)
    return loss

def make_gradient_loss(x, g_target, g_guess, without_last_bias, w_reg):
    assert len(g_target) == len(g_guess)    
    if without_last_bias:
        # remove bias term last layer from the optimization (it is always the last entry)
        a = collapse(g_target[:-1])
        b = collapse(g_guess[:-1])
    else:
        a = collapse(g_target)
        b = collapse(g_guess)
        
    reg = tf.reduce_mean(tf.image.total_variation(x)) * w_reg
    loss = 1 + tf.keras.losses.cosine_similarity(a, b) + reg
    return loss

def attack_iteration(model, y, x_, g_target, opt, without_last_bias, w_reg):    
    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as tape_inner:
            y_ = model(x_)
            loss_guess = make_loss(y_, y)
        g_guess = tape_inner.gradient(loss_guess, model.trainable_variables)

        loss_attack = make_gradient_loss(x_, g_target, g_guess, without_last_bias, w_reg)
    gradients_x = tape.gradient(loss_attack, x_)
    gradients_y = tape.gradient(loss_attack, y)

    opt.apply_gradients(zip([gradients_x, gradients_y], [x_, y]))

    return loss_attack

def gradient_inversion(
    model,
    gradient,
    opt,
    num_iter,
    n_targets,
    input_shape,
    class_num,
    without_last_bias,
    w_reg,
    SEED,
    x_range=(-1.,1.),
    num_v=10):
    print("Running inversion...")
    
    tf.random.set_seed(SEED)
    
    y_val = np.random.normal(size=(n_targets, class_num))
    y_var = tf.Variable(y_val, dtype=tf.float32, trainable=True)
    
    clip = lambda x: tf.clip_by_value(x, *x_range)
    x_val = np.random.normal(size=(n_targets, *input_shape))
    x_ = tf.Variable(x_val, dtype=tf.float32, trainable=True, constraint=clip)

    LOG_FQ = int(num_iter / num_v)

    logs = []
    for i in range(num_iter):
        loss = attack_iteration(model, y_var, x_, gradient, opt, without_last_bias, w_reg)
        if(i % LOG_FQ == 0):
            logs.append(loss.numpy())
            pg = int( (i / num_iter) * 100)
            print(f'\t[{pg}%] loss: {loss.numpy()}')
            
    return x_.numpy(), y_var.numpy(), logs


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
            print(f'\tlayer: {i} with shape {agg_i.shape} recovered?: {diff==0}')
            
    recovered = 1. - (diff / num_par)
    return recovered

