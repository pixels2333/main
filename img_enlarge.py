#img_enlarge
#该程序主要用于在划分好训练集与验证集后的图片进行数据扩容26倍（包括原始图片）


import os

import cv2
import numpy as np
from PIL import Image, ImageOps

# 用于图像处理与增强，实现将图片按照v值调整并导出

def rotate_and_flip(img):                                                         #图像数据的旋转与变换
    img90 = img.rotate(-90, expand=True)
    img180 = img90.rotate(180, expand=True)
    img270 = img180.rotate(90, expand=True)
    imgflr = ImageOps.mirror(img270)            #图片水平翻转
    imgfud = ImageOps.flip(imgflr)              #图片垂直翻转
                                                                         #采用字典的方式定义了一个orientation属性用于命名
    rotated_flipped_images = [
        {'image': img90, 'orientation': '90'},
        {'image': img180, 'orientation': '180'},
        {'image': img270, 'orientation': '270'},
        {'image': imgflr, 'orientation': 'flr'},  # 水平翻转
        {'image': imgfud, 'orientation': 'fud'}  # 垂直翻转
    ]

    return rotated_flipped_images


def adjust_hsv(img,v=1.0):                                      #由RGB转换到HSV再转换回RGB
    img = img.convert('RGB')
    img_np = np.array(img).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v, 0, 255).astype(np.uint8)       #调整v值
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
    adjusted_img = Image.fromarray(img_rgb)
    return adjusted_img

# 设置包含图片的文件夹路径
folder_path = 'T_V_img_data/train_183_4758'
v_values = [1.0, 0.8, 0.9, 1.1, 1.2]  # 为每个旋转和翻转后的图像提供不同的v值
Save_path = 'T_V_img_data/train_183_4758'


# 遍历文件夹内的所有文件
for filename in os.listdir(folder_path):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 使用PIL打开图片
    with Image.open(file_path) as img:
        imgs_rotated_flipped = rotate_and_flip(img)

        for img_info in imgs_rotated_flipped:                           #遍历字典
            img_rf = img_info['image']                                  #提取图片
            orientation = img_info['orientation']                     # 获取orientation信息

            # 应用不同的v值进行HSV调整
            for v in v_values:
                img_a = adjust_hsv(img_rf.copy(), v)  # 在调整v值时复制多份，以防止覆盖

                # 构建保存路径，不使用output_dir，直接使用Save_path
                base_filename, ext = os.path.splitext(filename)
                save_filename = f"{base_filename}_v{v}_{orientation}{ext}"
                save_path = os.path.join(Save_path, save_filename)

                # 保存图像
                img_a.save(save_path)



