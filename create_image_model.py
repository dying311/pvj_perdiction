"""
@filename:cmp_img.py
@author:dying
@time:2024-05-21
"""

import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras_preprocessing.image import load_img, img_to_array


data_dir = 'D:\Desktop\strom_prediction\strom\\'  # 定义风暴数据的目录路径
image_height = 366  # 图像高度
image_width = 366  # 图像宽度


#使用GPU运行程序，需要安装CUDA和CUDNN，以及对应版本的tensorflow-GPU
#若CPU运存足够可以忽略+-
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置第一个GPU的显存限制为6144MB（6GB）
        #自行分配
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)


# 读取数据
def load_train_image(storm_id, data_type):
    storm_path = os.path.join(data_dir, storm_id)
    train_images = []
    for file_name in sorted(os.listdir(storm_path)):
        if file_name.startswith(data_type):
            if file_name.endswith('.jpg'):
                # 读取图片文件
                image_path = os.path.join(storm_path, file_name)
                image = load_img(image_path, target_size=(image_height, image_width))
                image_array = img_to_array(image) / 255.0  # 归一化
                train_images.append(image_array)

    return np.array(train_images)


# 生成未来图像预测模型
def build_image_prediction_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(input_shape[0] * input_shape[1] * input_shape[2], activation='sigmoid'),
        layers.Reshape(input_shape)
    ])
    model.compile(optimizer='adam', loss='mse')  # 使用均方误差作为损失函数
    return model


def train_image_prediction_model(train_images):
    model = build_image_prediction_model(input_shape)
    model.fit(train_images, train_images, epochs=num_epochs, batch_size=batch_size, verbose=1)
    return model


# 加载训练数据
storm_id = input("请输入要读取的风暴ID: ")
train_images = load_train_image(storm_id, 'train')

# 定义模型参数
num_epochs = 30
batch_size = 16
input_shape = (366, 366, 3)

# 训练模型
print("\n训练模型一：生成未来图像预测")
image_prediction_model = train_image_prediction_model(train_images)

# 保存模型
image_prediction_model.save(f"{storm_id}_image_prediction_model_epoch{num_epochs}.h5")

print("\n模型一训练完成并已保存。")
