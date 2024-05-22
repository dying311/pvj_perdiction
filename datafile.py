"""
@filename:datafile.py
@author:dying
@time:2024-05-22
"""
import os

data_dir = 'D:\\Desktop\\strom_prediction\\strom\\'  # 定义风暴数据的目录路径
storm_name = input("请输入风暴名称")
storm_path = os.path.join(data_dir, storm_name)
files = os.listdir(storm_path)


def split_train_val_data(files, train_ratio=0.8):  # 确定训练集大小，训练集与测试集按4：1划分

    num_train_samples = int(len(files) * train_ratio)

    # 分割成训练集和验证集
    train_paths = files[:num_train_samples]
    test_paths = files[num_train_samples:]
    rename_file(train_paths, 'train')
    rename_file(test_paths, 'test')
    # 训练集加上前缀'train_'，测试集加上前缀'test_'
    return train_paths, test_paths


def rename_file(data_path, prefix):
    for data_name in data_path:
        new_data_name = prefix + '_' + data_name
        new_path = os.path.join(storm_path, new_data_name)
        old_path = os.path.join(storm_path, data_name)
        os.rename(old_path, new_path)


split_train_val_data(files)
