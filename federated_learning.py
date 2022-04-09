import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from utility import *

class FL_server:
    " Simulate FL up to the Secure Aggregation phase " 
    
    def __init__(self, users, model, suppressing_technique, target=0):
        self.users = users
        self.num_users = len(users)
        self.target = target
        self.model = model
        self.global_parameters = model.trainable_variables
        self.suppressing_technique = suppressing_technique
        
        self.gradients = [None] * self.num_users
        
    def models_distribution(self):
        suppressed_parameters = self.suppressing_technique(self.model)
        
        for i, user_i in enumerate(self.users):
            # Model Inconsistency
            if i == self.target:
                print(f"Sending honest parameters to user {i}")
                # target user. Send normal parameter
                parameters = clone_list_tensors(self.global_parameters)
            else:
                print(f"Sending tampered parameters to user {i}")
                parameters = clone_list_tensors(suppressed_parameters)
                
            user_i.set_model(parameters)
            
        print("End model distribution\n")
    
    def SA(self):
        # Virtual Secure Aggregation
        gradients = [None] * self.num_users

        for i, user_i in enumerate(self.users):
            print(f"Receiving and aggregating model update from user {i}")
            g = user_i.local_training()
            gradients[i] = g
            
        agg_model_update = sum_list_tensors(gradients)
        agg_model_update = [x.numpy() for x in agg_model_update]
        
        print("End aggregation\n")
        
        return agg_model_update
            

class FL_SGD_client_classification:
    " A FedSGD user "
    
    @staticmethod
    def loss(y, y_):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y, y_)
        loss = tf.reduce_mean(loss)
        return loss
        
    def __init__(self, model, training_set):
        self.model = model
        self.training_set = iter(training_set.repeat(-1))
        
        self.gradient = None
        self.data = None
        
    def set_model(self, parameters):
        assign_list_tensors(self.model.trainable_variables, parameters)
        
    def local_training(self):
        x, y = next(self.training_set)
        
        with tf.GradientTape() as tape:
            y_ = self.model(x)
            l = self.loss(y, y_)
        g = tape.gradient(l, self.model.trainable_variables)
        
        self.gradient = [x.numpy() for x in g]
        self.data = x.numpy(), y.numpy()
        
        return g
    
    
    
def setup_users_classification(User, num_users, model, global_dataset, local_training_set_size, batch_size, parse_x):
    X, Y = global_dataset
    assert len(X) == len(Y)
    nds = len(X)
    assert local_training_set_size * num_users <= nds
    
    X = parse_x(X)
    
    users = [None] * num_users
    for i in range(num_users):
        X_i = X[i*local_training_set_size:(i+1)*local_training_set_size] 
        Y_i = Y[i*local_training_set_size:(i+1)*local_training_set_size] 
        X_i = tf.data.Dataset.from_tensor_slices(X_i)
        Y_i = tf.data.Dataset.from_tensor_slices(Y_i)
    
        XY_i = tf.data.Dataset.zip((X_i, Y_i)).batch(batch_size)
        
        model_i = deepCopyModel(model)
        
        users[i] = User(model_i, XY_i)
        
    return users