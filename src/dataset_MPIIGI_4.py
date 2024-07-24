"""
针对MPIIGI数据集进行dataset,一共有三个或者四个人的数据，分开去提取三个或者四个人的数据
"""
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch import nn, optim
from ssi_stream_utils import Stream

feature_len = 2477
#2477/1453

modalities = [
    ".audio.egemapsv2.stream",
    ".audio.w2vbert2_embeddings.stream",  # 这个数据的采样率不一样，所以可以消融实验看看
    ".video.clip.stream",
    ".video.openface2.stream",
    ".video.openpose.stream",
]
# 自定义Dataset类，用于数据加载
class CustomDataset(Dataset):
    def __init__(self, data_dir, modalities, phase):

        self.data_dir = data_dir
        self.modalities = modalities  #模态名称列表
        self.data = []  #数据
        self.partner1_data = [] #同伴1数据
        self.partner2_data = []  # 同伴1数据
        self.partner3_data = []  # 同伴1数据
        self.labels = []  #数据标签
        self.data_label = np.empty((0, 4*feature_len+4)) #用来存储这一对novice和expert所有的特征和标签，注意每次清零
        #4956/2908
        self.phase = phase
        self.four_label_path()
        self.split_data_label()
        self.process_data()  #数据加载的方法

    # 存储所有成对标签的路径，返回anno字典：
    def four_label_path(self):

        anno = {}
        # 遍历训练目录以构建注释字典

        for path, sub_dirs, files in os.walk(self.data_dir):
            session_id = os.path.basename(path)  # path=r'/lijia/yuyangchen/data/MEE_2024/MPIIGI/008和其他文件夹'
            key = "pose1_2_3_4;" + session_id  # 就是每对的关键字，用来标识，如：pose_1_2_3_4；/yuyangchen/data/MEE_2024/MPIIGI/008

            pose1_path = ""
            pose2_path = ""
            pose3_path = ""
            pose4_path = ""

            for f in files:
                if "1.engagement.annotation" in f:  # 提取pose1_label文件
                    pose1_path = os.path.join(path, f)
                if "2.engagement.annotation" in f:  # 提取pose2_label文件
                    pose2_path = os.path.join(path, f)
                if "3.engagement.annotation" in f:  # 提取pose3_label文件
                    pose3_path = os.path.join(path, f)
                if "4.engagement.annotation" in f:  # 提取pose4_label文件
                    pose4_path = os.path.join(path, f)

            #会有一个多余的空值，通过判断给除掉：
            if pose1_path != "":
                anno[key] = [pose1_path, pose2_path, pose3_path, pose4_path]
        return anno

    def split_data_label(self): #很呆的一个注释：第一个参数需要是self
        """
        对特征进行切分，切分的方法是有重叠的，每个窗口大小为96，而步长为32，核心长度也是32
        如： [-32, 63], [0, 95], [31, 127]以此类推
        并且将这些特征进行保存，当再次加载的时候，其实就相当于一个在滑动的窗口了，窗口大小为96，步长为32
        """
        step = 32 #滑动窗口的步长是32
        length = 96 #滑动窗口的大小是96 [extended_len:core_len:extended_len]

        data_label_len = self.data_label.shape[0]

        # 确定按core_len切分之后的sample数量,core_len和step相等：
        if (data_label_len) % step != 0:
            n_samples = data_label_len // step + 1
        else:
            n_samples = data_label_len // step

        #对data_label进行切割：
        for i in range(n_samples):
            start = i * step - step
            end = start + length
            if start < 0:  # 在原来数据上加了前面的空白部分
                zeros_y = np.zeros((step, 4*feature_len+4))
                #4956/2908
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, 4*feature_len+4))
                #4956/2908
                padded_data_label = np.concatenate((self.data_label[start:, :], zeros_y))
                split_data_label = padded_data_label
            else:
                split_data_label = self.data_label[start:end, :]

            #对split_data_label进行切片，分别得到novice_data, expert_data, novice_label, expert_label：
            pose1_data = split_data_label[:, :feature_len]
            pose2_data = split_data_label[:, feature_len:2*feature_len]
            pose3_data = split_data_label[:, 2*feature_len:3*feature_len]
            pose4_data = split_data_label[:, 3*feature_len:4*feature_len]

            pose1_label = split_data_label[:, -4:-3].squeeze()
            pose2_label = split_data_label[:, -3:-2].squeeze()
            pose3_label = split_data_label[:, -2:-1].squeeze()
            pose4_label = split_data_label[:, -1:].squeeze()


        #将这些data和label进行排序组合，变成（data, partner1_data, partner2_data, partner3_data, partner4_data, label）的形式;

            #第一个样本：
            self.data.append(pose1_data)  #数据
            self.partner1_data.append(pose2_data) #同伴1数据
            self.partner2_data.append(pose3_data)  # 同伴2数据
            self.partner3_data.append(pose4_data)  # 同伴3数据
            self.labels.append(pose1_label)  #数据标签
            #第二个样本：
            self.data.append(pose2_data)  # 数据
            self.partner1_data.append(pose3_data)  # 同伴1数据
            self.partner2_data.append(pose4_data)  # 同伴2数据
            self.partner3_data.append(pose1_data)  # 同伴3数据
            self.labels.append(pose2_label)  # 数据标签
            # 第三个样本：
            self.data.append(pose3_data)  # 数据
            self.partner1_data.append(pose4_data)  # 同伴1数据
            self.partner2_data.append(pose1_data)  # 同伴2数据
            self.partner3_data.append(pose2_data)  # 同伴3数据
            self.labels.append(pose3_label)  # 数据标签
            # 第四个样本：
            self.data.append(pose4_data)  # 数据
            self.partner1_data.append(pose1_data)  # 同伴1数据
            self.partner2_data.append(pose2_data)  # 同伴2数据
            self.partner3_data.append(pose3_data)  # 同伴3数据
            self.labels.append(pose4_label)  # 数据标签



    def process_data(self):

        anno_dict = self.four_label_path()

        # #进行检查一对输出有没有问题：
        # key = "novice_expert;050"
        # anno = {}
        # anno[key] = anno_dict[key]

        # 使用 tqdm 包装迭代器，显示进度条
        # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
        for entry in tqdm(anno_dict, desc="Loading " +self.phase+ " data"):
            self.data_label = np.empty((0, 4*feature_len+4)) #用来存储这一对novice和expert所有的特征和标签
            #4956/2908
            pose1_features = {}
            pose2_features = {}
            pose3_features = {}
            pose4_features = {}

            lengths = [] #把每个pose的所有特征的长度都记录下来

            pose1_path = anno_dict[entry][0]
            pose2_path = anno_dict[entry][1]
            pose3_path = anno_dict[entry][2]
            pose4_path = anno_dict[entry][3]
            base_path = os.path.dirname(pose1_path)

            # 对于每个pose的 label 文件，找到其对应的特征文件，并且读取其特征,分别保存在novice_features和expert_features：
            for modality in self.modalities:
                pose1_modality_file = os.path.join(base_path, "subjectPos1" + modality)
                pose2_modality_file = os.path.join(base_path, "subjectPos2" + modality)
                pose3_modality_file = os.path.join(base_path, "subjectPos3" + modality)
                pose4_modality_file = os.path.join(base_path, "subjectPos4" + modality)

                pose1_stream_f = Stream().load(pose1_modality_file)
                pose2_stream_f = Stream().load(pose2_modality_file)
                pose3_stream_f = Stream().load(pose3_modality_file)
                pose4_stream_f = Stream().load(pose4_modality_file)

                pose1_features[modality] = pose1_stream_f.data
                pose2_features[modality] = pose2_stream_f.data
                pose3_features[modality] = pose3_stream_f.data
                pose4_features[modality] = pose4_stream_f.data

                lengths.append(pose1_stream_f.data.shape[0])
                lengths.append(pose2_stream_f.data.shape[0])
                lengths.append(pose3_stream_f.data.shape[0])
                lengths.append(pose4_stream_f.data.shape[0])

            # 读取novice和expert label 文件中的内容，
            with open(pose1_path, "r") as f:
                pose1_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            with open(pose2_path, "r") as f:
                pose2_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            with open(pose3_path, "r") as f:
                pose3_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            with open(pose4_path, "r") as f:
                pose4_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            #长度列表加上novice和expert label的长度：
            lengths.append(len(pose1_label))
            lengths.append(len(pose2_label))
            lengths.append(len(pose3_label))
            lengths.append(len(pose4_label))

            #至此所有长度都得到了，取最小值，保证novice和expert所有的特征和标签数量全部对齐：
            num_samples = min(lengths)


            #对novice label做相应处理(去 NaN)并转化为 float:
            pose1_values = []

            for label in pose1_label:
                if label == '-nan(ind)':
                    pose1_values.append(float(0))
                else:
                    pose1_values.append(float(label))
            pose1_values = np.nan_to_num(pose1_values)

            #对novice label做相应处理(去 NaN)并转化为 float:
            pose2_values = []

            for label in pose2_label:
                if label == '-nan(ind)':
                    pose2_values.append(float(0))
                else:
                    pose2_values.append(float(label))
            pose2_values = np.nan_to_num(pose2_values)

            #对novice label做相应处理(去 NaN)并转化为 float:
            pose3_values = []

            for label in pose3_label:
                if label == '-nan(ind)':
                    pose3_values.append(float(0))
                else:
                    pose3_values.append(float(label))
            pose3_values = np.nan_to_num(pose3_values)

            #对novice label做相应处理(去 NaN)并转化为 float:
            pose4_values = []

            for label in pose4_label:
                if label == '-nan(ind)':
                    pose4_values.append(float(0))
                else:
                    pose4_values.append(float(label))
            pose4_values = np.nan_to_num(pose4_values)


#
# 下面注释部分的代码运行很慢，需要一个小时处理一个文件，从以下几点进行优化处理： (优化之后10秒就能结束处理，所以还是很重要的，主要是不能一直循环vstack,太浪费时间）
#         1.使用批量操作，代替循环，就是一直循环着去vstack
#         2.减少初始化判断操作，就是一直if
#         3.预先分配内存，这样就不用加一次申请一次，加一次申请一次了（预申请内存也很重要）
#
            # 分别将novice和expert的所有特征合并，并且与 label 做对齐：
            # for i in tqdm(range(num_samples)):
            #     novice_sample = np.concatenate([np.nan_to_num(novice_features[modality][i]) for modality in self.modalities])
            #     expert_sample = np.concatenate([np.nan_to_num(expert_features[modality][i]) for modality in self.modalities])
            #
            #     #这里的总维度为 2477+2477+1+1 = 4956
            #     ne_data_label = np.concatenate((novice_sample, expert_sample, [novice_values[i],expert_values[i]]))
            #     ne_data_label = np.array(ne_data_label).reshape(1, -1) #保证在堆叠的时候，两个都是二维数组
            #     if self.data_label.shape[0] == 0:
            #         self.data_label = ne_data_label  # 初次赋值
            #     else:
            #         self.data_label = np.vstack((self.data_label, ne_data_label))
            #     #注意为什么这里用vstack而不用append:
            #     #       1.vstack专门为垂直堆叠而设计，所以更加高效
            #     #       2.append会默认将数组全部压平然后进行堆叠，所以对堆叠形成二维数组不有效

            self.data_label = np.zeros((num_samples, 4*feature_len+4))
            for i in range(num_samples):
                pose1_sample = np.concatenate([np.nan_to_num(pose1_features[modality][i]) for modality in self.modalities])
                pose2_sample = np.concatenate([np.nan_to_num(pose2_features[modality][i]) for modality in self.modalities])
                pose3_sample = np.concatenate([np.nan_to_num(pose3_features[modality][i]) for modality in self.modalities])
                pose4_sample = np.concatenate([np.nan_to_num(pose4_features[modality][i]) for modality in self.modalities])
                #这里的总维度为 2477+2477+2477+2477+1+1+1+1 = 4956
                ne_data_label = np.concatenate((pose1_sample, pose2_sample, pose3_sample, pose4_sample,[pose1_values[i], pose2_values[i], pose3_values[i], pose4_values[i]]))
                self.data_label[i] = ne_data_label

            self.split_data_label()

            # print(np.array(self.data).shape)
            # print(np.array(self.partner_data).shape)
            # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), np.array(self.partner1_data[idx]), np.array(self.partner2_data[idx]), np.array(self.partner3_data[idx]), np.array(self.labels[idx])




class OnlineDataset(Dataset):
    def __init__(self, dataset):
        """
        dataset: A list of tuples (data, partner_data, label)
        """
        self.dataset = dataset
        self.data_len = len(dataset)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data, partner1_data, partner2_data,partner3_data,label = self.dataset[idx]
        return (torch.tensor(data, dtype=torch.float32),
                torch.tensor(partner1_data, dtype=torch.float32),
                torch.tensor(partner2_data, dtype=torch.float32),
                torch.tensor(partner3_data, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


'''mpiigi_train_dataset = CustomDataset("../mpiigi/", modalities, "mpiigi")
train_dataset = OnlineDataset(mpiigi_train_dataset)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                             pin_memory=True)

fixed_values = [
    0.291667, 0.333333, 0.375000, 0.416667,
    0.458333, 0.500000, 0.541667, 0.583333,
    0.625000, 0.666667, 0.708333, 0.750000,
    0.791667, 0.833333, 0.875000, 0.916667
]
# 定义映射规则
mapping = {value: idx for idx, value in enumerate(fixed_values)}


# 函数用于找到最接近的映射值
def find_closest_value(value, fixed_values):
    closest_value = min(fixed_values, key=lambda x: abs(x - value))
    return closest_value


# 增加对不在列表中的连续值处理情况，将其映射为最接近的类别
def map_labels(labels, mapping, fixed_values):
    mapped_labels = torch.empty_like(labels, dtype=torch.long)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            value = labels[i, j].item()
            closest_value = find_closest_value(value, fixed_values)
            mapped_labels[i, j] = mapping[closest_value]

    return mapped_labels


# 应用映射
for i, data in enumerate(trainloader, 0):
    inputs, partnet_inputs1,partnet_inputs2,partnet_inputs3, labels = data
    print("原始",labels)
    mapped_labels = map_labels(labels, mapping, fixed_values)
    print("映射后",mapped_labels)'''

import concurrent.futures
from math import ceil
# 初始化数据集
'''mpiigi_train_dataset = CustomDataset("../mpiigi/", modalities, "mpiigi")
train_dataset = OnlineDataset(mpiigi_train_dataset)
trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                             pin_memory=True)
print(len(train_dataset))

modalities_dim = [88, 1024, 512, 714, 139]

def split_features(inputs, modalities_dim):
    batchsize, timesteps, total_dim = inputs.shape
    assert total_dim == sum(modalities_dim), "总维度与各个特征维度之和不匹配"

    split_features = []
    start_idx = 0
    for dim in modalities_dim:
        split_features.append(inputs[:, :, start_idx:start_idx + dim])
        start_idx += dim

    return split_features

def process_batch(data):
    inputs, partnet_inputs1, partnet_inputs2,partnet_inputs3,labels = data
    split_inputs = split_features(inputs, modalities_dim)
    #第五个特征就改为4，其他同理
    feature_np = split_inputs[4].detach().cpu().numpy().astype(np.float32).flatten()  # 转换为 float32
    return feature_np

# 初始化直方图参数
bins = 50
hist_data = np.zeros(bins, dtype=np.float32)
global_max = -np.inf
global_min = np.inf

# 逐批处理数据并更新直方图
for data in tqdm(trainloader, desc="处理数据中"):
    feature_np = process_batch(data)

    # 更新全局最大和最小值
    batch_max = np.max(feature_np)
    batch_min = np.min(feature_np)
    if batch_max > global_max:
        global_max = batch_max
    if batch_min < global_min:
        global_min = batch_min

    # 计算当前批次的直方图
    hist, _ = np.histogram(feature_np, bins=bins, range=(global_min, global_max))
    hist_data += hist

# 打印全局最大值和最小值
print(f'最大值: {global_max}')
print(f'最小值: {global_min}')

# 绘制直方图
bin_edges = np.linspace(global_min, global_max, bins + 1)
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist_data, width=np.diff(bin_edges), alpha=0.75, color='blue', edgecolor='black')
plt.title('Feature 4 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# 保存图表
plt.savefig('feature_4_distribution.png')
plt.close()
'''
