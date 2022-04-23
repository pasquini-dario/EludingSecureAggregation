import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from functools import partial
import h5py

img_size = 32
buffer_size = 1000

PATH_tinyimagenet = '/home/pasquini/STORAGE/DeepLearningDatasets/tiny-imagenet-200/tinyImagenet_nolabelforval.hdf5'

@tf.function
def data_aug(x):
    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_crop(x, [img_size, img_size, 3])
    return x

@tf.function
def std_parse_img(batch, resize=False, da=False):
    x = batch['image']
    if resize:
        x = tf.image.resize(x, (img_size, img_size))
    if da:
        x = data_aug(x)
    x = tf.cast(x, tf.float32)
    x = (x / (255/2) - 1)
    y = batch['label']
    return x, y

def get_one_sample(dataset, buffer_size=buffer_size):
    # it gets a single element from a dataset
    dataset = iter(dataset.shuffle(buffer_size).take(1))
    x, y = next(dataset)

    if len(x.shape) == 4:
        x = x[0]
        y = y[0]
        
    return x[None, :, :, :].numpy(), y.numpy()

def load_dataset_classification(key, batch_size, split='train', repeat=-1, da=False):
    
    if key in ['cifar10', 'cifar100']:
        ds, info = tfds.load(
            key,
            split=split,
            shuffle_files=True,
            with_info=True,
            download=True
        )
        try:
            num_class = info.features['label'].num_classes
        except:
            num_class = 2
        #x_shape = info.features['image'].shape  
        
    if key in ['tinyimagenet']:
        path = PATH_tinyimagenet
        with h5py.File(path, 'r') as f:
            x = f['X_'+split][:]
            y = f['Y_'+split][:]
            x = tf.data.Dataset.from_tensor_slices(x)
            y = tf.data.Dataset.from_tensor_slices(y)
            ds = tf.data.Dataset.zip( {'image':x, 'label':y})
            
            num_class = 200
        
    x_shape = (img_size, img_size, 3)
    
    resize = False
    if key in ['tinyimagenet']:
        resize = True
        
    parse_img = partial(std_parse_img, resize=resize, da=da)      
    
    ds = ds.map(parse_img)
    ds = ds.shuffle(buffer_size)
    ds = ds.repeat(repeat)
    ds = ds.batch(batch_size, drop_remainder=True)
    
    return ds, x_shape, num_class