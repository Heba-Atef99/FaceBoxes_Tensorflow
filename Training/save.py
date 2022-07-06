import tensorflow as tf
import json
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


CONFIG = '/content/drive/MyDrive/FaceBoxes_Model/Training_Code/config.json'
OUTPUT_FOLDER = '/content/drive/MyDrive/FaceBoxes_Model/Training_Code/models/export/run00'
GPU_TO_USE = '0'

WIDTH, HEIGHT = None, None
# size of an input image,
# set (None, None) if you want inference
# for images of variable size


tf.compat.v1.logging.set_verbosity('INFO')
params = json.load(open(CONFIG))
model_params = params['model_params']

config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config
)
estimator = tf.compat.v1.estimator.Estimator(model_fn, params=model_params, config=run_config)


def serving_input_receiver_fn():
    images = tf.compat.v1.placeholder(dtype=tf.uint8, shape=[None, HEIGHT, WIDTH, 3], name='image_tensor')
    features = {'images': tf.cast(images, tf.float32)*(1.0/255.0)}
    return tf.compat.v1.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)
