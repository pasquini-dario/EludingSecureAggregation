import tensorflow as tf
import numpy as np
import tqdm 

import models
from datasets import load_dataset_classification, get_one_sample
from canary_utility import *

def load_dataset(
    dataset_key,
    dataset_key_shadow,
    batch_size_test,
    batch_size_train,
    data_aug_shadow=False
):
    # load target for MIA 
    pool_targets, x_shape, class_num = load_dataset_classification(dataset_key, batch_size_test, split='test')
    # load local training set users for evaluation phase
    validation, _, _ = load_dataset_classification(dataset_key, batch_size_test, split='train', repeat=1)
    # load shadow dataset for canary injection
    shadow, _, _ = load_dataset_classification(dataset_key_shadow, batch_size_train, split='train', da=data_aug_shadow)
    

    x_target, y_target = get_one_sample(pool_targets)
    return validation, shadow,  x_shape, class_num, (x_target, y_target)

def setup_model(
    model_id,
    canary_id,
    x_shape,
    class_num,
):
    make_model = models.models[model_id]
    model = make_model(x_shape, class_num)
    
    layer_idx, g_canary_shift, kernel_idx = models.canaries[model_id][canary_id]
    
    model = repack_model(model, layer_idx, kernel_idx)
    pre_canary_layer_trainable_variables = get_preCanaryTrainable_variables_Conv2D(model, layer_idx+1)

    return model, layer_idx, g_canary_shift, kernel_idx, pre_canary_layer_trainable_variables



@tf.function
def make_loss(att, mask, W):
    s = att.shape[1] * att.shape[2]
    att = tf.reshape(att, (-1, s))
    mask = tf.reshape(mask, (-1, s))
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)(mask, att)
    loss = loss * W
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def attack_iteration(model, x, mask, W, variables, opt):   
    with tf.GradientTape() as tape:
        _, att = model(x, training=True)    
        loss = make_loss(att, mask, W)  
        
    gradients = tape.gradient(loss, variables)
    opt.apply_gradients(zip(gradients, variables))
    return loss


def inject_canary(
    max_number_of_iters,
    batch_size,
    model,
    target,
    shadow_dataset,
    variables,
    opt,
    loss_threshold=0.0010,
    check_steps=10,
    min_num_iterations=500,
    w=5,
):
    LOG = []

    canary_shape = model.output[1].shape.as_list()[1:]
    class_num = model.output[0].shape[1]
    
    mask = np.ones((batch_size, *canary_shape), np.float32) * 0
    mask[-1] = 1

    mask_b = np.ones((batch_size, 1), np.float32)
    mask_b[-1] = (batch_size - 1) * w

    loss_avg = 0.
    for i, batch in tqdm.tqdm(enumerate(shadow_dataset)):
        x, _ = batch
        
        x = tf.concat([x[:-1], target], 0)
        loss = attack_iteration(model, x, mask, mask_b, variables, opt)
        loss = loss.numpy()
        
        loss_avg += loss
        
        if i % check_steps == 0:
            loss_avg /= check_steps
            LOG.append(loss)
            
            if loss_avg <= loss_threshold and i > min_num_iterations:
                print("Loss Threshold reached!")
                return LOG, True
                
            loss_avg = 0.

        if i > max_number_of_iters:
            print("Max number of iterations reached!")
            return LOG, False

    return LOG, False

############################################################################################

def find_fail(X, acts):
    Xfail = []
    for i, act in enumerate(acts):
        failed = np.any(act > 0)
        if failed:
            Xfail.append(X[i])
    return Xfail


def evaluate_canary_attack(
    model,
    dataset_validation,
    target,
    variables,
    loss_function,
    g_canary_shift=-1,
    kernel_idx=0,
    max_num_batches_eval=None
):
    
    # tn fp
    neg = [0, 0]
    # tp fn
    pos = [0, 0]
    
    failed = []

    n = 0
    for i, batch in enumerate(dataset_validation):
        # batch without target
        negative, y = batch
        # batch with target (the label does not care)
        positive = tf.concat([negative[:-1], target], 0)

        neg_g, neg_act = get_gradient(negative, y, model, loss_function, variables)
        neg_g = get_canary_gradient(neg_g, g_canary_shift, kernel_idx).sum().tolist()
        
        pos_g, pos_act = get_gradient(positive, y, model, loss_function, variables)
        pos_g = get_canary_gradient(pos_g, g_canary_shift, kernel_idx).sum().tolist()

        neg[neg_g != 0] += 1
        pos[pos_g == 0] += 1
        
        if neg_g != 0:
            fail = find_fail(negative.numpy(), neg_act)
            failed.append(fail)
        
        n += 1

        if max_num_batches_eval and i >= max_num_batches_eval:
            print("Max number of iterations evaluation reached!")
            break

    acc = (neg[0] + pos[0]) / (n * 2)
    recall = pos[0] / n
    try:
        precision = pos[0] / (pos[0] + neg[1])
    except:
        precision = np.nan
    
    out = {
        'accuracy' : acc,
        'recall' : recall,
        'precision' : precision,
    }
    
    return out, failed