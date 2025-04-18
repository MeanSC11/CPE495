import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.python.client import device_lib
print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))
print("\n=== All Devices ===")
for device in device_lib.list_local_devices():
    print(f"{device.name} - {device.device_type} - {device.physical_device_desc}")

print(tf.__version__)

#tf.debugging.set_log_device_placement(True)
#
#with tf.device('/GPU:0'):
#    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
#   c = tf.matmul(a, b)
#    print(c)