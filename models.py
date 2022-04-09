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

# def resnet38(input_shape, num_classes, batch_norm=tf.keras.layers.LayerNormalization, activation='relu'):
#     xin = tf.keras.layers.Input(input_shape)
    
#     x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(xin)
#     x = tf.keras.layers.Activation(activation)(x)
#     x = tf.keras.layers.MaxPool2D(2)(x)    
#     x = ResBlock(x, 64, batch_norm=batch_norm, activation=activation)
#     x = ResBlock(x, 64, batch_norm=batch_norm, activation=activation)
#     x = ResBlock(x, 128, batch_norm=batch_norm, reduce=2, activation=activation)
#     x = ResBlock(x, 128, batch_norm=batch_norm, activation=activation)
#     x = ResBlock(x, 128, batch_norm=batch_norm, activation=activation)
#     x = ResBlock(x, 256, batch_norm=batch_norm, reduce=2, activation=activation)
#     x = ResBlock(x, 256, batch_norm=batch_norm, activation=activation)
#     x = ResBlock(x, 256, batch_norm=batch_norm, activation=activation)
#     x = tf.keras.layers.Flatten()(x)
#     logits = tf.keras.layers.Dense(num_classes)(x)
#     y_ = tf.nn.softmax(logits)
    
#     return tf.keras.Model(xin, y_) 


# def load_model(key):
#     models = {
#                 'ResNet50_scratch' : (lambda num_class, **args: tf.keras.applications.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=num_class, classifier_activation="softmax"), parse_img),
#                 'ResNet50_imagenet' : (lambda num_class, **args: tf.keras.applications.ResNet50(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax"), parse_img),
        
#                 'InceptionV3_imagenet' : (lambda num_class, **args: tf.keras.applications.InceptionV3(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax"), parse_img),
        
#                 'ResNet18' : (lambda num_class, input_shape, **args: resnet18(input_shape, num_class, **args), parse_img)
#     }
#     return models[key]

def sparse_classification_loss(y, y_):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y, y_)
    loss = tf.reduce_mean(loss)
    return loss

def accuracy(y, y_):
    y_ = tf.argmax(y_, 1)
    b = y == y_
    b = tf.cast(b, tf.float32)
    return tf.reduce_mean(b)