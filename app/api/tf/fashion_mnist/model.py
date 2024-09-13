import os
import time
import keras as k
import numpy as np
from PIL import Image


def predict(url, cd):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    path_keras = cd + '../../../training/fashion_mnist/models/keras/k_model.keras'
    model = k.models.load_model(path_keras)
    pm = k.Sequential([model, k.layers.Softmax()])

    img = get_log_img(url, cd)
    p = pm.predict(img)
    p = np.argmax(p)
    p = class_names[p]
    print(p)
    return p


def get_log_img(url, cd):
    name = str(time.clock_gettime_ns(time.CLOCK_REALTIME)) + ' -- ' + url.split('/')[-1]
    image_path = k.utils.get_file(name, origin=url)
    img = Image.open(image_path).convert('L').resize((28,28))
    t = np.array(img)

    Image.fromarray(t.astype('uint8')).save(cd + '/predicted_images/' + name)

    return np.reshape(t, (1,28,28))


if __name__ == '__main__':
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    predict(url, os.getcwd())
