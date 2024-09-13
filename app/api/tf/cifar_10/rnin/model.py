import os
import time
import tensorflow as tf
import keras as k
import numpy as np
from PIL import Image


def predict(url, cd):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model_dir = cd + '../../../../training/cifar_10/models/resnet_imagenet/model'
    model = tf.saved_model.load(model_dir)
    model_fn = model.signatures['serving_default']

    img = get_log_img(url, cd)
    p = model_fn(img)['logits']
    p = np.argmax(p)
    p = class_names[p]
    print(p)
    return p


def get_log_img(url, cd):
    name = str(time.clock_gettime_ns(time.CLOCK_REALTIME)) + ' -- ' + url.split('/')[-1]
    image_path = k.utils.get_file(name, origin=url)
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    t = np.array(img)

    Image.fromarray(t.astype('uint8')).save(cd + '/predicted_images/' + name)

    r = tf.convert_to_tensor(t)

    return r[tf.newaxis, ...]


if __name__ == '__main__':
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    predict(url, os.getcwd())
