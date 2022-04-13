import tensorflow as tf

rng_seed = 1

batch_size_train = 256
batch_size_test = 64
batch_size_tests = [32, 64, 128, 256, 512]
max_num_batches_eval = 150
data_aug_shadow = False

# 0 MSE, 1 BinaryCrossEntropy
injection_type = 1
if injection_type == 0:
    pos_w = 1
    loss_threshold = 0.0001
elif injection_type == 1:
    pos_w = 5
    loss_threshold = 0.0003

# if pretrain model before canary injection (simulating FL)
pre_train = False

model_id = 'resnet20'
canary_id = 'last_layer'
max_number_of_iters = 2000

# FedAVG
num_iter_fedAVG = 15
learning_rate_fedAVG = 0.001

# attack
max_number_of_iters = 10000
#init_lr = .01
#setps_lrs = [1000, 2000]
#opt = tf.keras.optimizers.Adam(lr_schlr(init_lr, setps_lrs))
opt = tf.keras.optimizers.Adam()
