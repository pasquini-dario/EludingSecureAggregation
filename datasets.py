import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

@tf.function
def parse_img(batch):
    x = batch['image']
    x = tf.cast(x, tf.float32)
    x = (x / (255/2) - 1)
    y = batch['label']
    return x, y

def get_one_sample(dataset, buffer_size=10000):
    # it gets a single element from a dataset
    dataset = iter(dataset.shuffle(buffer_size).take(1))
    x, y = next(dataset)

    if len(x.shape) == 4:
        x = x[0]
        y = y[0]
        
    return x[None, :, :, :].numpy(), y.numpy()

def load_dataset_classification(key, batch_size, split='train', repeat=-1):
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
    x_shape = info.features['image'].shape    
    ds = ds.batch(batch_size, drop_remainder=True).repeat(repeat)
    
    if key == 'cifar10' or key == 'cifar100':
        ds = ds.map(parse_img)
    
    return ds, x_shape, num_class