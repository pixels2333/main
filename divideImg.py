# divideImg程序
# 主要功能为将划分好的train_val图像按照8：2进行随机划分并保留中间的划分结果存入两个新的文件夹中，接下来将使用数据增强扩容程序

import os
import random
import shutil

# 设定原始图片文件夹路径
original_folder = 'Training and validation'

# 设定新的训练集和测试集文件夹路径
train_folder = 'T_V_img_data/train'
val_folder = 'T_V_img_data/val'

# 确保新的训练集和测试集文件夹存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 获取所有图片的完整路径
images = [os.path.join(original_folder, f) for f in os.listdir(
    original_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 打乱图片顺序
random.shuffle(images)

# 计算分割点
split_index = int(0.8 * len(images))

# 分割图片列表
train_images = images[:split_index]
test_images = images[split_index:]

# 将图片复制到新的文件夹
for image_path in train_images:
    filename = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(train_folder, filename))

for image_path in test_images:
    filename = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(val_folder, filename))

print(
    f'Images have been split and copied into {train_folder} and {val_folder} with an 8:2 ratio.')

print("图像和数据处理完成。")
