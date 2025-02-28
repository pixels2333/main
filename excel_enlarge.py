# excel_enlarge
# 该程序用于同步图片数据集扩增后，实现excel数据集扩增


import pandas as pd
import os
import re

# 定义一个函数来复制数据


def copy_data(df, original_name, new_name):
    # 找到原始图片名称对应的行
    original_rows = df[df['image'].astype(str) == original_name].copy()
    if not original_rows.empty:
        # 复制整行数据
        new_row = original_rows.iloc[0].copy()
        # 更新图片名称
        new_row['image'] = new_name
        return new_row
    return None


# 读取Excel文件
excel_path = 'T_V_img_data/val_enlarge.xlsx'  # Excel文件路径
df_excel = pd.read_excel(excel_path)

# 获取Excel中已有的图片名称集合
existing_images = set(df_excel['image'].astype(str))

# 图片文件夹路径
image_folder_path = 'T_V_img_data/val_46_1196'  # 图片文件夹路径

# 用于存储需要添加的新图片名称列表
new_images_to_add = []

# 遍历文件夹中的所有文件
for filename in os.listdir(image_folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        # 使用正则表达式搜索基本的图片名称
        match = re.search(r'(\d+-\d+)', filename)
        if match:
            base_name = match.group(0)  # 直接使用匹配的整个字符串作为基本名称
            original_image_name = f"{base_name}.jpg"  # 构造原始图片名称
            # 检查原始图片名称是否已存在于Excel中
            if original_image_name in existing_images:
                # 构造新图片名称
                new_image_name = filename
                # 检查新图片名称是否已存在于Excel中
                if new_image_name not in existing_images:
                    # 如果不存在，添加到新图片列表
                    new_images_to_add.append((new_image_name, base_name))

# 遍历需要添加的图片名称列表
for new_image, base_name in new_images_to_add:
    # 构造原始图片名称
    original_image_name = f"{base_name}.jpg"
    # 使用copy_data函数复制数据
    new_data = copy_data(df_excel, original_image_name, new_image)
    if new_data is not None:
        # 将复制的数据添加到DataFrame中
        df_excel = pd.concat(
            [df_excel, pd.DataFrame([new_data])], ignore_index=True)

# 打印新添加的图片数量
print(f"Added {len(new_images_to_add)} new images to the DataFrame.")

# 将更新后的DataFrame写回到Excel文件
df_excel.to_excel(excel_path, index=False)
