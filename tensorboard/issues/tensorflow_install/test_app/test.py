import keras as k
import datetime

def main():

    mnist = k.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    def create_model():
        return k.models.Sequential([
            k.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
            k.layers.Dense(512, activation='relu', name='layers_dense'),
            k.layers.Dropout(0.2, name='layers_dropout'),
            k.layers.Dense(10, activation='softmax', name='layers_dense_2')
        ])

    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=x_train,
              y=y_train,
              epochs=5,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])


if __name__ == '__main__':
    main()