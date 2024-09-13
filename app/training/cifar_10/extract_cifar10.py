import os
import numpy as np
import keras as k
from PIL import Image


def main(dataset, class_names, target, type):
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()
    save_data(
        train_data,
        np.ndarray.flatten(train_labels),
        test_data,
        np.ndarray.flatten(test_labels),
        class_names,
        target,
        type)


def save_data(t_data, t_labels, v_data, v_labels, class_names, target, type):
    l = target + '/{0:05d}-{1}-{2}' + type

    k = 0
    i = 0
    for img in t_data:
        im = Image.fromarray(img.astype('uint8'))
        im.save(l.format(k, class_names[t_labels[i]], 'train'))
        k += 1
        i += 1
    i = 0
    for img in v_data:
        im = Image.fromarray(img.astype('uint8'))
        im.save(l.format(k, class_names[v_labels[i]], 'test'))
        k += 1
        i += 1


if __name__ == '__main__':
    dataset = k.datasets.cifar10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    target = os.getcwd() + '/images'
    os.mkdir(target)
    main(dataset, class_names, target, '.png')
