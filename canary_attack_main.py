import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import partial
import os, sys
import importlib

import myPickle

import models
from utility import lr_schlr

from canary_utility import local_training
from datasets import load_dataset_classification

home_output = './results/'
loss_function = models.sparse_classification_loss

if __name__ == '__main__':
    try:
        setting = sys.argv[1]
        id = int(sys.argv[2])
    except:
        print("[USAGE] setting_file id")
        sys.exit(1)    
        
    output = []
        
    C = importlib.import_module(setting)
    rng_seed = C.rng_seed + id
    tf.random.set_seed(rng_seed)
    np.random.seed(rng_seed)    
            
    output.append(rng_seed)
    
    if C.injection_type == 1:
        from canary_attack import load_dataset, setup_model, evaluate_canary_attack, inject_canary

    name = '_'.join(map(str,[C.dataset_key, C.dataset_key_shadow, C.injection_type, C.pos_w, C.batch_size_train, C.loss_threshold, C.model_id, C.canary_id, C.learning_rate_fedAVG]))
    name = f'{id}-{name}'
    output_path = os.path.join(home_output, name)
    print(name)
    
    # load datasets
    validation, shadow, x_shape, class_num, (x_target, y_target) = load_dataset(
        C.dataset_key,
        C.dataset_key_shadow,
        C.batch_size_test,
        C.batch_size_train,
        data_aug_shadow=C.data_aug_shadow,
    )
    
    output.append(x_target)
    
    # load model and pick canary location
    model, layer_idx, g_canary_shift, kernel_idx, pre_canary_layer_trainable_variables = setup_model(
        C.model_id,
        C.canary_id,
        x_shape,
        class_num
    )    

    print("Injecting canary ...")
    inj_logs, ths_reached = inject_canary(
        C.max_number_of_iters,
        C.batch_size_train,
        model,
        x_target,
        shadow,
        pre_canary_layer_trainable_variables,
        C.opt,
        loss_threshold=C.loss_threshold,
        w=C.pos_w,
    )
    
    if not ths_reached:
        print("Canary injection failed! Try again.")
        sys.exit(1)
    
    output.append(inj_logs)
    
    # prepare evaluation function
    test_canary_fn = partial(
        evaluate_canary_attack,
        target=x_target,
        variables=pre_canary_layer_trainable_variables,
        loss_function=loss_function,
        g_canary_shift=g_canary_shift,
        kernel_idx=kernel_idx,
        max_num_batches_eval=C.max_num_batches_eval
    )
    
    print("Evaluation FedSGD ....")
    scores_FedSGD = []
    for sgd_batch_size_evaluation in C.batch_size_tests:
        print(f"\tEvaluation FedSGD - batch size: {sgd_batch_size_evaluation} ... ")
        validation_i, _, _ = load_dataset_classification(C.dataset_key, sgd_batch_size_evaluation, split='train', repeat=1)
        score_FedSGD, failed_FedSGD = test_canary_fn(model, validation_i)
        print(sgd_batch_size_evaluation, score_FedSGD)
        scores_FedSGD.append( (sgd_batch_size_evaluation, (score_FedSGD, failed_FedSGD)) )
        
    output.append(scores_FedSGD)
        
    print("Evaluation FedAVG ....")
    canary_scores_FedAVG = local_training(
        model,
        validation,
        C.num_iter_fedAVG,
        C.learning_rate_fedAVG,
        loss_function,
        test_canary_fn
    )
    output.append(canary_scores_FedAVG)
    
    myPickle.dump(output_path, output)
