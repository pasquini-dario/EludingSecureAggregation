import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np


NORM = tf.keras.layers.LayerNormalization
def plain_res_block(x_in, filters, stride, bn):
    
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x_in)
    
    if bn:
        x = NORM()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(x)
    
    if bn:
        x = NORM()(x)
    x = layers.Activation("relu")(x)
    
    if stride > 1:
        x_in = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(x_in)
        
        if bn:
            x_in = NORM()(x_in)   
    
    x = x + x_in
    
    return x

def resnet20(input_shape, num_classes, bn=True, activation='relu'):
    x_in = layers.Input(input_shape)
    
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", use_bias=False)(x_in)
   
    if bn:
        x = NORM()(x)
    x = layers.Activation("relu")(x)
    
    x = plain_res_block(x, 16, 1, bn)
    x = plain_res_block(x, 16, 1, bn)
    x = plain_res_block(x, 16, 1, bn)
    
    x = plain_res_block(x, 32, 2, bn)
    x = plain_res_block(x, 32, 1, bn)
    x = plain_res_block(x, 32, 1, bn)
    
    x = plain_res_block(x, 64, 2, bn)
    x = plain_res_block(x, 64, 1, bn)
    x = plain_res_block(x, 64, 1, bn)
    x = layers.AveragePooling2D(pool_size=8)(x)
        
    x = layers.Flatten()(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    y_ = tf.nn.softmax(logits)
    
    return tf.keras.Model(x_in, y_)


def sparse_classification_loss(y, y_):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y, y_)
    loss = tf.reduce_mean(loss)
    return loss

def accuracy(y, y_):
    y_ = tf.argmax(y_, 1)
    b = y == y_
    b = tf.cast(b, tf.float32)
    return tf.reduce_mean(b)


models = {
    'resnet20':resnet20,
}

canaries = {
    'resnet20':{
        'last_layer':(68, -2, 0)
    },
    
}

