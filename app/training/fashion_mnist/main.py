import tensorflow as tf
import keras as k
import numpy as np
import os
import datetime


def main():
    cwd = os.getcwd()

    path_models_keras = './models/keras/'
    path_models_keras_file = path_models_keras + 'k_model.keras'
    path_logs = cwd + '../../../../tensorboard/tensorflow_logs/fashion_mnist/'

    if os.path.exists(path_models_keras_file):
        model = k.models.load_model(path_models_keras_file)
    else:
        model = k.Sequential([
            k.layers.Flatten(input_shape=(28, 28)),
            k.layers.Dense(128, activation='relu'),
            k.layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                      loss=k.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    train_data, train_labels, test_data, test_labels, class_names = get_data()
    model = train_save_model(model,
                             train_data, train_labels,
                             test_data, test_labels,
                             path_models_keras, path_models_keras_file,
                             path_logs)

    loaded = k.models.load_model(path_models_keras_file)

    comp_predictions(model, loaded, test_data)


def train_save_model(model,
                     train_data, train_labels,
                     test_data, test_labels,
                     path_models_keras, path_models_keras_file,
                     path_logs):

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(train_data, train_labels,
              epochs=10,
              validation_data=(test_data, test_labels),
              callbacks=[
                  k.callbacks.TensorBoard(path_logs + timestamp, histogram_freq=1)
              ])

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    model.save(path_models_keras + '/fashion_mnist-{0}-{1:.5f}-{2:.5f}.keras'.format(timestamp, test_acc, test_loss))
    model.save(path_models_keras_file)

    return model


def get_data():
    dataset = k.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names


def comp_predictions(model, loaded, data):
    pm = k.Sequential([model, k.layers.Softmax()])
    pl = k.Sequential([loaded, k.layers.Softmax()])

    predictions = []
    for i in range(12):
        predictions.append(np.argmax(pm.predict(data[i][tf.newaxis, ...])))
    print(predictions)

    predictions = []
    for i in range(12):
        predictions.append(np.argmax(pl.predict(data[i][tf.newaxis, ...])))
    print(predictions)


if __name__ == '__main__':
    main()
