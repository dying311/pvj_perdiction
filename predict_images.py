import os
import matplotlib
import numpy as np
from keras import models
from keras.losses import MeanSquaredError
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

data_dir = 'D:\\Desktop\\strom_prediction\\strom\\'  # 定义风暴数据的目录路径
image_height = 366  # 图像高度
image_width = 366  # 图像宽度

matplotlib.use('TkAgg')


# 加载数据
def load_single_dataset(storm_id, data_type):
    storm_path = os.path.join(data_dir, storm_id)
    storm_images = []
    for file_name in sorted(os.listdir(storm_path)):
        if file_name.startswith(data_type):
            if file_name.endswith('.jpg'):
                # 读取图片文件
                image_path = os.path.join(storm_path, file_name)
                image = load_img(image_path, target_size=(image_height, image_width))
                image_array = img_to_array(image) / 255.0  # 归一化
                storm_images.append(image_array)

    return np.array(storm_images)


# 加载测试数据
storm_id = input("请输入要读取的风暴ID: ")
test_images = load_single_dataset(storm_id, "test")

# 加载模型
print("加载模型")
model_path = os.path.join(storm_id + f"_image_prediction_model_epoch30.h5")#epoch自行修改为训练轮次
image_prediction_model = models.load_model(model_path, compile=False)
image_prediction_model.compile(optimizer='adam', loss=MeanSquaredError())

# 预测未来图像
future_images = image_prediction_model.predict(test_images)

# 显示三张预测结果
for i, image in enumerate(future_images[:3]):
    plt.imshow(image)
    plt.title(f"Image {i + 1} Prediction")
    plt.axis('off')
    plt.figure(i - 1)
    plt.show()
