import keras as k
import numpy as np
import os
import datetime
import time


def main():
    timestamp_start = time.time()

    cwd = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keras_models = './models/keras/l4/'
    keras_model = keras_models + 'k_model.keras'
    path_logs = cwd + '../../../../tensorboard/tensorflow_logs/cifar_10/l4/'

    dataset = k.datasets.cifar10
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()

    if os.path.exists(keras_model):
        model = load_model(keras_model)
    else:
        model = define_model()

    model, test_loss, test_acc = train_model(model,
                                             train_data, train_labels,
                                             test_data, test_labels,
                                             path_logs,
                                             timestamp)

    save_model(model, test_loss, test_acc, keras_models, keras_model, timestamp)

    loaded = load_model(keras_model)

    pm = softmax_model(model)
    pl = softmax_model(loaded)

    dataset = test_data[0:15]
    print_predictions(pm.predict(dataset))
    print_predictions(pl.predict(dataset))

    timestamp_end = time.time()
    print("Elapsed time: " + str(timestamp_end - timestamp_start))


def define_model():
    model = k.Sequential([
        k.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
        k.layers.BatchNormalization(),
        k.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.MaxPooling2D(pool_size=(2,2)),
        k.layers.Dropout(0.3),

        k.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.MaxPooling2D(pool_size=(2,2)),
        k.layers.Dropout(0.5),

        k.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.MaxPooling2D(pool_size=(2,2)),
        k.layers.Dropout(0.5),

        k.layers.Flatten(),
        k.layers.Dense(128, activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.Dropout(0.5),
        k.layers.Dense(10, activation='softmax')    # num_classes = 10
    ])
    model.compile(optimizer='adam',
                  loss=k.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def load_model(model_path):
    return k.models.load_model(model_path)


def train_model(model, train_data, train_labels, test_data, test_labels, path_logs, timestamp):
    model.fit(train_data, train_labels,
              epochs=10,
              validation_data=(test_data, test_labels),
              callbacks=[
                  k.callbacks.TensorBoard(path_logs + timestamp, histogram_freq=1)
              ])
    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    return model, test_loss, test_acc


def save_model(model,test_loss, test_acc, directory, file, timestamp):
    model.save(file)
    model.save(directory + '/cifar10-{0}-{1:.5f}-{2:.5f}.keras'.format(timestamp, test_acc, test_loss))


def softmax_model(model):
    return k.Sequential([model, k.layers.Softmax()])


def print_predictions(predictions):
    results = []
    for p in predictions:
        results.append(np.argmax(p))
    print(results)


if __name__ == '__main__':
    main()
