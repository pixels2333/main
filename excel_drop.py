#excel_drop程序
#主要用于将数据按照图片进行提取并写入新的数据集


import os
import pandas as pd
from openpyxl import Workbook

# 定义数据和图像的目录
excel_path = 'Field measurements.xlsx'  # Excel文件路径，这里使用您提供的文件名
image_dir = 'T_V_img_data/val_46_1196'  # 图片文件夹路径
output_dir = 'T_V_img_data'  # 输出文件夹路径

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取Excel文件
df = pd.read_excel(excel_path)

# 获取图像文件列表
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 确保图像列表是有序的，如果不需要可以删除此行

# 创建一个新的Excel工作簿
workbook = Workbook()
sheet = workbook.active
sheet.title = 'Image Data'

# 写入表头
headers = ['image', 'LFW', 'LDW', 'LA']
sheet.append(headers)

# 遍历图像文件列表
for image_file in image_files:
    # 在Excel数据中查找匹配的行
    match = df[df['image'] == image_file]

    # 如果找到匹配项，则写入Excel
    if not match.empty:
        for _, row in match.iterrows():
            sheet.append([row['image'], row['LFW'], row['LDW'], row['LA']])
    else:
        print(f"警告：未在Excel中找到图片 {image_file} 的匹配项")

# 保存新的Excel文件
output_excel_path = os.path.join(output_dir, 'val.xlsx')
workbook.save(output_excel_path)
print(f"数据已写入 {output_excel_path}")