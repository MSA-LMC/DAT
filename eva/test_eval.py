#读取预测文件和真实标签文件，然后计算评分，看看结果如何：

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metric import concordance_correlation_coefficient
import pandas as pd

# 存储所有成对标签的路径，返回anno字典：
def couple_label_path(data_dir):
    anno = []
    # 获取所有目录和文件路径并在最终字典里面排序,这样能够保证所有的数据在测试时与最终标签能够对齐：
    for path, sub_dirs, files in os.walk(data_dir):
        if files:
            anno.append(path)  #这里的path是特征文件的文件夹路径如：'/dev/shm/yyc_data/Data/Noxi/val/007'
    anno.sort() #对文件夹进行排序，这样可以保证最后的标签顺序可控

    # print(anno)

    return anno


#自定义数据标签的读取：
def read_label(label_dir, role):

    #按序遍历label文件夹，然后读取所有文件：
    if role == 'novice':
        file_label = os.path.join(label_dir, "novice.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "novice.engagement.annotation_pred.csv")
    else:
        file_label = os.path.join(label_dir, "expert.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "expert.engagement.annotation_pred.csv")

    # 读取 label 文件中的内容:
    label_values = read_label_file(file_label)
    pred_values = read_label_file(file_pred)

    return label_values, pred_values


#定义函数进行label文件的数据读取：
def read_label_file(file_name):

    with open(file_name, "r") as f:
        label = np.genfromtxt(f, delimiter="\n", dtype=str)

    #预先申请内存进行label值的保存：
    #如果一个文件中的label不足32的倍数，就将其补全,保证与训练时一致：
    if len(label) % 32 != 0:
        label_len = (len(label)//32 +1) * 32

    label_values = np.zeros((label_len, ))
    for i, label_value in enumerate(label):
        if label_value == '-nan(ind)':
            label_values[i] = float(0)
        else:
            label_values[i] = float(label_value)

    #去除nan值：
    label_values = np.nan_to_num(label_values)
    # print(label_values.shape)

    return label_values


anno = couple_label_path("/dev/shm/yyc_data/Data/Noxi/val/")

role_list = ["expert","novice"]

label_values = np.empty((0,))
pred_values = np.empty((0,))

for i in tqdm(anno, desc="Loading: "):
    for j in role_list:
        label_value, pred_value = read_label(i,j)
        label_values = np.concatenate(label_values, label_value)
        pred_values = np.concatenate(pred_values, pred_value)

ccc = concordance_correlation_coefficient(label_values, pred_values)
print("pred_average: ", ccc)