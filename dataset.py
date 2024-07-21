"""
实现两步走的dataset实现策略，与一步走的dataset实现策略，看看哪种方法更好：
评测结果：
当不能将idx精确到一个文件时，还是两步走比较好，将不需要的变量清空，然后使得内存管理更友好一些
"""
import gc

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch import nn, optim
from ssi_stream_utils import Stream

feature_len = 2477
#2477/1453



# 自定义Dataset类，用于数据加载
class CustomDataset(Dataset):
    def __init__(self, data_dir, modalities, phase):

        self.data_dir = data_dir
        self.modalities = modalities  #模态名称列表
        self.data = []  #数据
        self.partner_data = [] #同伴数据
        self.labels = []  #数据标签
        self.data_label = np.empty((0, 2*feature_len+2)) #用来存储这一对novice和expert所有的特征和标签，注意每次清零
        #4956/2908
        self.phase = phase
        # self.couple_label_path()
        # self.split_data_label()
        self.process_data()  #数据加载的方法

    # 存储所有成对标签的路径，返回anno字典：
    def couple_label_path(self):

        anno = {}
        # 遍历训练目录以构建注释字典
        for path, sub_dirs, files in os.walk(self.data_dir):
            session_id = os.path.basename(path)  # path=r'/yuyangchen/data/MEE_2024/Noxi/train\001和其他文件夹'
            key = "novice_expert;" + session_id  # 就是每对的关键字，用来标识，如：novice_expert；001

            novice_path = ""
            expert_path = ""

            for f in files:
                if "novice.engagement.annotation" in f:  # 提取novice_label文件
                    novice_path = os.path.join(path, f)
                if "expert.engagement.annotation" in f:  # 提取expert_label文件
                    expert_path = os.path.join(path, f)

            #会有一个多余的空值，通过判断给除掉：
            if novice_path != "":
                anno[key] = [novice_path, expert_path]

        return anno

    def split_data_label(self, role): #很呆的一个注释：第一个参数需要是self
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
                zeros_y = np.zeros((step, 2*feature_len+2))
                #4956/2908
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, 2*feature_len+2))
                #4956/2908
                padded_data_label = np.concatenate((self.data_label[start:, :], zeros_y))
                split_data_label = padded_data_label
            else:
                split_data_label = self.data_label[start:end, :]

            #对split_data_label进行切片，分别得到novice_data, expert_data, novice_label, expert_label：
            novice_data = split_data_label[:, :feature_len]
            #2477/1453
            expert_data = split_data_label[:, feature_len:feature_len*2]
            #2477，4954/1453，2906
            novice_label = split_data_label[:, -2:-1].squeeze()
            expert_label = split_data_label[:, -1:].squeeze()


        #将这些data和label进行排序组合，变成（data, partner_data, label）的形式;

            if role == 'novice':
                #第一个样本(新手样本）：
                self.data.append(novice_data)  #数据
                self.partner_data.append(expert_data) #同伴数据
                self.labels.append(novice_label)  #数据标签

            else:
                #第二个样本(专家样本）：
                self.data.append(expert_data)  #数据
                self.partner_data.append(novice_data) #同伴数据
                self.labels.append(expert_label)  #数据标签



    def process_data(self):

        anno_dict = self.couple_label_path()

        # 使用 tqdm 包装迭代器，显示进度条
        # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
        for entry in tqdm(anno_dict, desc="Loading " +self.phase+ " data"):
            self.data_label = np.empty((0, 2*feature_len+2)) #用来存储这一对novice和expert所有的特征和标签
            #4956/2908
            novice_features = {}
            expert_features = {}
            lengths = [] #把novice和expert所有的长度都记录下来

            novice_path = anno_dict[entry][0]
            expert_path = anno_dict[entry][1]
            base_path = os.path.dirname(novice_path)

            # 对于每个novice和expert label 文件，找到其对应的特征文件，并且读取其特征,分别保存在novice_features和expert_features：
            for modality in self.modalities:
                novice_modality_file = os.path.join(base_path, "novice" + modality)
                expert_modality_file = os.path.join(base_path, "expert" + modality)
                novice_stream_f = Stream().load(novice_modality_file)
                expert_stream_f = Stream().load(expert_modality_file)
                novice_features[modality] = novice_stream_f.data
                expert_features[modality] = expert_stream_f.data
                lengths.append(novice_stream_f.data.shape[0])
                lengths.append(expert_stream_f.data.shape[0])

            # 读取novice和expert label 文件中的内容，
            with open(novice_path, "r") as f:
                novice_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            with open(expert_path, "r") as f:
                expert_label = np.genfromtxt(f, delimiter="\n", dtype=str)

            #长度列表加上novice和expert label的长度：
            lengths.append(len(novice_label))
            lengths.append(len(expert_label))

            #至此所有长度都得到了，取最小值，保证novice和expert所有的特征和标签数量全部对齐：
            num_samples = min(lengths)


            #对novice label做相应处理(去 NaN)并转化为 float:
            novice_values = []

            for label in novice_label:
                if label == '-nan(ind)':
                    novice_values.append(float(0))
                else:
                    novice_values.append(float(label))
            novice_values = np.nan_to_num(novice_values)

            # 对expert label做相应处理(去 NaN)并转化为 float:
            expert_values = []

            for label in expert_label:
                if label == '-nan(ind)':
                    expert_values.append(float(0))
                else:
                    expert_values.append(float(label))
            expert_values = np.nan_to_num(expert_values)


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

            self.data_label = np.zeros((num_samples, 2*feature_len+2))
            for i in range(num_samples):
                novice_sample = np.concatenate([np.nan_to_num(novice_features[modality][i]) for modality in self.modalities])
                expert_sample = np.concatenate([np.nan_to_num(expert_features[modality][i]) for modality in self.modalities])
                #这里的总维度为 2477+2477+1+1 = 4956
                ne_data_label = np.concatenate((novice_sample, expert_sample, [novice_values[i], expert_values[i]]))
                self.data_label[i] = ne_data_label

            self.split_data_label(role='expert')
            self.split_data_label(role='novice')


            # print(np.array(self.data).shape)
            # print(np.array(self.partner_data).shape)
            # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), np.array(self.partner_data[idx]), np.array(self.labels[idx])




class OnlineDataset(Dataset):
    def __init__(self, dataset):
        """
        dataset: A list of tuples (data, partner_data, label)

        将数据都放在__init__中进行处理，这样子初始化之后，后续的数据索引，就可以直接获得数据，而不需要再进行数据处理浪费时间了：
        """

        self.dataset = dataset  # 原始数据源
        # self.data = []
        # self.partner_data = []
        # self.labels = []
        #注意： 第二个dataset,如果还要再init里面进行数据的处理那就需要更多的内存空间，但是会更快，当内存管理没有做到那么细致的时候，建议还是不要再init里面进行数据处理，放到getitem中进行：
        # # Apply normalization transform during initialization
        # for data, partner_data, label in tqdm(dataset, desc="normalizing: "):
        #     data_normalized = normalize_transform(torch.tensor(data, dtype=torch.float32))
        #     partner_data_normalized = normalize_transform(torch.tensor(partner_data, dtype=torch.float32))
        #     self.data.append(data_normalized)
        #     self.partner_data.append(partner_data_normalized)
        #     self.labels.append(torch.tensor(label, dtype=torch.float32))

        self.data_len = len(self.dataset)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # 获取原始数据
        data, partner_data, label = self.dataset[idx]

        # 进行归一化
        data_normalized = torch.tensor(data, dtype=torch.float32)
        partner_data_normalized = torch.tensor(partner_data, dtype=torch.float32)

        return data_normalized, partner_data_normalized, (torch.tensor(label, dtype=torch.float32))


'''modalities = [
        ".audio.egemapsv2.stream",
        ".audio.w2vbert2_embeddings.stream",  
        ".video.clip.stream",
        ".video.openface2.stream",
        ".video.openpose.stream",
    ]
train_dataset = CustomDataset("/data/public_datasets/MEE_2024/Noxi/val", modalities, "Noxi_train")
trainloader = DataLoader(train_dataset, batch_size=128,num_workers=6,prefetch_factor=6,shuffle=True)
for i, data in enumerate(trainloader, 0):
    inputs, partnet_inputs, labels = data
    print(inputs.size())
    print(labels.size())'''



# class HDF5Dataset(Dataset):
#     def __init__(self, hdf5_file_path, sample_frac=1.0):
#         self.hdf5_file_path = hdf5_file_path
#
#         #定义样本载入的比例：
#         self.sample_frac = sample_frac
#
#         with h5py.File(self.hdf5_file_path, 'r') as f:
#             self.data = f['data'][:]
#             self.data_len = f['data'].shape[0]
#             self.partner_data = f['partner_data'][:]
#             self.labels = f['labels'][:]
#
#             if self.sample_frac<1.0:
#                 sampled_indices = np.random.choice(self.data_len, int(self.data_len * self.sample_frac), replace=False)
#                 self.data = self.data[sampled_indices]
#                 self.partner_data = self.partner_data[sampled_indices]
#                 self.labels = self.labels[sampled_indices]
#                 self.data_len = len(self.data)
#
#     def __len__(self):
#         return self.data_len
#
#     def __getitem__(self, idx):
#         data = self.data[idx][:]
#         partner_data = self.partner_data[idx][:]
#         label = self.labels[idx][:]
#         return torch.tensor(data, dtype=torch.float32), torch.tensor(partner_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

