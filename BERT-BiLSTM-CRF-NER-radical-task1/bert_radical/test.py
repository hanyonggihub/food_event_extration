import tensorflow as tf
import os
print(tf.__version__)
b = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
print(b)
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0)) # 返回GPU名称，设备索引默认从0开始

print(torch.cuda.current_device())  # 返回当前设备索引