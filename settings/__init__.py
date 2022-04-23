import tensorflow as tf

rng_seed = 1

batch_size_train = 256
batch_size_test = 64
batch_size_tests = [32, 64, 128, 256, 512]
max_num_batches_eval = 150
data_aug_shadow = False

injection_type = 1
if injection_type == 1:
    pos_w = 5
    loss_threshold = 0.0003
else:
    raise NotImplementedError()


model_id = 'resnet20'
canary_id = 'last_layer'
max_number_of_iters = 3000

# FedAVG
num_iter_fedAVG = 15
learning_rate_fedAVG = 0.001

# attack
max_number_of_iters = 10000
opt = tf.keras.optimizers.Adam()
