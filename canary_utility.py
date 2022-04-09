import tensorflow as tf
import numpy as np


def repack_model(model, layer_idx, kernel_idx):
    canary_layer = model.layers[layer_idx]
    print("---->", canary_layer.name, canary_layer.output.shape)
    canary_layer = canary_layer.output[:, :, :, kernel_idx]
    _model = tf.keras.Model([model.input], [model.output, canary_layer])
    return _model


def get_preCanaryTrainable_variables_Conv2D(model, layer_idx):
    # get trainable variables before canary kernel
    pre_canary_layer_trainable_variables = []
    
    for l in model.layers[:layer_idx]:
        var = l.trainable_variables
        print(l.name)
        for v in var:
            print('\t', v.name)
        print()
        pre_canary_layer_trainable_variables += var
        
    return pre_canary_layer_trainable_variables
    
def enumerate_layes(model):
    for i, l in enumerate(model.layers):
        print(i, '\t', l.name, l.output.shape)
        
def get_canary_gradient(G, g_canary_shift, kernel_idx):
    g = G[g_canary_shift]
    if len(g.shape) == 1:
        return g[kernel_idx]
    elif len(g.shape) == 4:
        return g[:, :, :, kernel_idx]
              
def get_gradient(x, y, model, loss_function, variables):
    with tf.GradientTape() as tape:
        y_, att = model(x, training=True)
        loss = loss_function(y, y_)
    g = tape.gradient(loss, variables)
    return [gg.numpy() for gg in g], att