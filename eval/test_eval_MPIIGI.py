#读取预测文件和真实标签文件，然后计算评分，看看结果如何：

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metric import concordance_correlation_coefficient
import pandas as pd
import matplotlib.pyplot as plt

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

#自定义数据标签的读取：
def read_label(label_dir, role):

    #按序遍历label文件夹，然后读取所有文件：
    if role == 'subjectPos1':
        file_label = os.path.join(label_dir, "subjectPos1.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "subjectPos1.engagement.annotation_pred.csv")
    elif role == "subjectPos2":
        file_label = os.path.join(label_dir, "subjectPos2.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "subjectPos2.engagement.annotation_pred.csv")
    elif role == "subjectPos3":
        file_label = os.path.join(label_dir, "subjectPos3.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "subjectPos3.engagement.annotation_pred.csv")
    else:
        file_label = os.path.join(label_dir, "subjectPos4.engagement.annotation.csv")
        file_pred = os.path.join(label_dir, "subjectPos4.engagement.annotation_pred.csv")


    # 读取 label 文件中的内容:
    label_values = read_label_file(file_label)
    pred_values = read_label_file(file_pred)

    return label_values, pred_values


#定义函数进行label文件的数据读取：
def read_label_file(file_name):

    with open(file_name, "r") as f:
        label = np.genfromtxt(f, delimiter="\n", dtype=str)

    label_len = len(label)

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


anno = couple_label_path("/home/u2023110082/MPIIGI/val/")
anno_selected = anno
# index = [0,1,0,0,1,1,0,0,0,1]
# for i, num in enumerate(index):
#     if num==1:
#         anno_selected.append(anno[i])
print(anno_selected)
#
role_list = ["subjectPos1", "subjectPos2", "subjectPos3", "subjectPos4"]
#
label_values = np.empty((0,))
pred_values = np.empty((0,))

for i in tqdm(anno_selected, desc="Loading: "):
    for j in role_list:
        label_value, pred_value = read_label(i,j)
        pred_value = pred_value[:len(label_value)]
        label_value = label_value[:len(pred_value)]
        label_values = np.concatenate((label_values, label_value))
        pred_values = np.concatenate((pred_values, pred_value))

ccc = concordance_correlation_coefficient(label_values, pred_values)
print("pred_average: ", ccc)

# type_list = ["label", "pred"]
#
# for label_dir in anno:
#     for role in role_list:
#         for type in type_list:
#             score = 1
#             print("select dir: ", label_dir)
#
#             label_value, pred_value = read_label(label_dir, role)
#
#             if type == "pred":
#                 pred_value = pred_value[:len(label_value)]
#                 label_value = label_value[:len(pred_value)]
#
#                 ccc = concordance_correlation_coefficient(label_value, pred_value)
#                 score = ccc
#
#                 print("pred_average: ", ccc)
#             else:
#                 pred_value = label_value
#
#
#             # 绘制数值图并添加评分到标题
#             plt.figure(figsize=(12, 6))
#             plt.plot(pred_value)
#             plt.title(f'Predicted Values {type} (Score: {score})')
#             plt.xlabel('Index')
#             plt.ylabel('Value')
#             plt.grid(True)
#             plt.savefig(f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/img/predicted_values_with_score_plot_{os.path.basename(label_dir)}_{role}_{type}.png')
