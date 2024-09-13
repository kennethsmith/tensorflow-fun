# https://www.tensorflow.org/tutorials/keras/classification
# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np
import json
import os
import time

from official.vision.serving import export_saved_model_lib


def main():
    timestamp_start = time.time()

    model_name = 'resnet_imagenet'
    dataset_name = 'cifar10'
    wd = os.getcwd() + '/models/' + model_name
    train_dir = wd + '/train'
    model_dir = wd + '/model'
    eval_file = wd + '/eval_logs.json'

    model = train_model(model_name, dataset_name, train_dir, model_dir, eval_file)

    dataset = tfds.load(dataset_name, split='test').batch(128).take(1)

    # Import and predict
    imported = tf.saved_model.load(model_dir)
    imported_fn = imported.signatures['serving_default']
    for data in dataset:
        predictions = []
        for image in data['image']:
            predictions.append(np.argmax(imported_fn(image[tf.newaxis, ...])['logits']))
        print(predictions)

    timestamp_end = time.time()
    print("Elapsed time: " + str(timestamp_end - timestamp_start))


def train_model(model_name, dataset_name, train_dir, model_dir, eval_file):

    exp_config = tfm.core.exp_factory.get_exp_config(model_name)
    tfds_name = dataset_name
    ds,ds_info = tfds.load(tfds_name, with_info=True)

    # # Adjust the model and dataset configurations so that it works with Cifar-10 (cifar10).
    # Configure model
    exp_config.task.model.num_classes = 10
    exp_config.task.model.input_size = list(ds_info.features["image"].shape)
    exp_config.task.model.backbone.resnet.model_id = 18

    # Configure training and testing data
    batch_size = 1024 # default 4098

    exp_config.task.train_data.input_path = ''
    exp_config.task.train_data.tfds_name = tfds_name
    exp_config.task.train_data.tfds_split = 'train'
    exp_config.task.train_data.global_batch_size = batch_size

    exp_config.task.validation_data.input_path = ''
    exp_config.task.validation_data.tfds_name = tfds_name
    exp_config.task.validation_data.tfds_split = 'test'
    exp_config.task.validation_data.global_batch_size = batch_size

    # # Adjust the trainer configuration.
    train_steps, exp_config.trainer.steps_per_loop, distribution_strategy = get_steps_and_distro_strat()

    exp_config.trainer.summary_interval = 100
    exp_config.trainer.checkpoint_interval = train_steps
    exp_config.trainer.validation_interval = 1000
    exp_config.trainer.validation_steps = ds_info.splits['test'].num_examples // batch_size
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100

    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=train_dir)

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=train_dir,
        run_post_eval=True)

    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[32, 32],
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(train_dir),
        export_dir=model_dir)

    write_json_file(eval_file, eval_logs)

    for images, labels in task.build_inputs(exp_config.task.validation_data).take(1):
        predictions = model.predict(images)
        predictions = tf.argmax(predictions, axis=-1)
        ps = []
        for v in predictions.numpy():
            ps.append(v)
        print(ps)

    return model


def get_steps_and_distro_strat():
    # # Adjust the trainer configuration.
    logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]
    print(logical_device_names)

    if 'TPU' in ''.join(logical_device_names):
        print('This may be broken in Colab.')
        tf.tpu.experimental.initialize_tpu_system()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
        return 5000, 100, tf.distribute.experimental.TPUStrategy(tpu)
    elif 'GPU' in ''.join(logical_device_names):
        print('This may be broken in Colab.')
        return 1000, 20, tf.distribute.MirroredStrategy()
    else:
        print('Running on CPU is slow, so only train for a few steps.')
        print('Warning: this will be really slow.')
        return 20, 5, tf.distribute.OneDeviceStrategy(logical_device_names[0])


def write_json_file(target, data):
    class NumpyTypeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(target, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyTypeEncoder)


if __name__ == '__main__':
    main()
