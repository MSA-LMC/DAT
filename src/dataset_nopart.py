import torch
from tqdm import tqdm  # 导入 tqdm 库
import numpy as np
import os
from ssi_stream_utils import Stream
from torch.utils.data import DataLoader, TensorDataset, Dataset
from config import *

"""
Mpii 的dataset,没有使用part
"""

# 自定义Dataset类，用于数据加载
class CustomDataset(Dataset):
    def __init__(self, data_dir, modalities,modalities_dim):
        self.data_dir = data_dir  #数据路径
        self.modalities = modalities   #模态名称列表
        self.feature_dim = sum(modalities_dim)  #特征维度

        self.labels = [] # 标签
        self.data = []  #数据   
        # 2477是特征的维度，1是标签的维度
        self.data_label = [] # 保存所有 数据 和 标签，用于后续的切分

        self.process_data()  #数据加载的方法
        self.split_data_Label()  #数据切分的方法

    #存储所有标签的路径，返回 anno字典
    def lable_path(self,file_dir):
        anno = {}
        # 遍历训练目录以构建注释字典
        for path, sub_dirs, files in os.walk(file_dir):
            for f in files:
                session_id = os.path.basename(path)  #path=r'D:\Data\MPIIGI\008和其他文件夹'
                role = f.split('.')[0] #role就是每个人的特征的名字，如subjectPos1.audio.egemapsv2
                key = role + ";" + session_id #就是每个特征的关键字，用来简化路径，如：subjectPos1.audio.egemapsv2；008
                file_keywords = ".engagement.annotation" #文件的关键字，如：subjectPos1.engagement.annotation
                if file_keywords in f: #提取.stream~后缀的文件，然后加入train_anno中
                    anno[key] = os.path.join(path, f)  # 简化路径连接
        return anno

    def process_data(self):
        # 保存所有标签路径
        anno_dict = self.lable_path(self.data_dir)  #标签路径

        # 使用 tqdm 包装迭代器，显示进度条
        for entry in tqdm(anno_dict, desc="Loading data"):
            # 创建一个特征字典
            features_each_role = {}
            # values 用来存储 label 的值
            values_each_role = []
            # lengths 总长度不用改，只是用来记录每个特征的长度
            lengths = []
            # base_path 是特征文件的路径,不用改
            base_path = os.path.dirname(anno_dict[entry])

            # 对于每个 label 文件，找到其对应的特征文件，并且读取其特征：
            # 并且读取其标签
            role = entry.split(";")[0]
            for modality in self.modalities:
                # 为了支持读取 text 编码后的npy文件
                if modality.split('.')[-1] == 'npy':
                    modality_file = os.path.join(base_path, role + modality)
                    stream_f_data = np.load(modality_file)
                    features_each_role[modality] = stream_f_data
                    lengths.append(stream_f_data.shape[0])
                elif modality.split('.')[-1] == 'stream':
                    modality_file = os.path.join(base_path, role + modality)
                    stream_f = Stream().load(modality_file)
                    features_each_role[modality] = stream_f.data
                    lengths.append(stream_f.data.shape[0])
                
            # 读取 label 文件中的内容，并做相应处理(去 NaN)转化为 float:
            label_file = os.path.join(base_path, role + ".engagement.annotation.csv")
            with open(label_file, "r") as f:
                anno_file = np.genfromtxt(f, delimiter="\n", dtype=str)
            lengths.append(len(anno_file))
            # 这里考虑对齐了吗？
            num_samples = min(lengths)
            # values 用来存储 label 的值
            # 下一段有必要吗？ 数据中有nan？
            for label in anno_file:
                if label == '-nan(ind)':
                    values_each_role.append(float(0))
                else:
                    values_each_role.append(float(label))
            values_each_role = np.nan_to_num(values_each_role)

            # 将特征合并，并且与 label 做对齐：
            for i in range(num_samples):
                sample_each_role = np.concatenate([np.nan_to_num(features_each_role[modality][i]) for modality in self.modalities])
                label_each_role = np.array([values_each_role[i]])
                temp_data_label = np.concatenate((sample_each_role, label_each_role))
                self.data_label.append(temp_data_label)
    # 对数据进行切分，需要预留参数，以便后期调整
    def split_data_Label(self):
        """
        对特征进行切分，切分的方法是有重叠的，每个窗口大小为96，而步长为32，核心长度也是32
        如： [-32, 63], [0, 95], [31, 127]以此类推
        并且将这些特征进行保存，当再次加载的时候，其实就相当于一个在滑动的窗口了，窗口大小为96，步长为32
        """
        step = core_length
        length = core_length + extended_length * 2

        self.data_label = np.array(self.data_label) #转换为numpy数组

        data_label_len = self.data_label.shape[0]

        # 确定按core_len切分之后的sample数量,core_len和step相等：
        if (data_label_len) % step != 0:
            n_samples = data_label_len // step + 1
        else:
            n_samples = data_label_len // step
        #对data和label进行切割：
        for i in tqdm(range(n_samples), desc="Spliting data"):
            start = i * step - step
            end = start + length
            if start < 0:  # 在原来数据上加了前面的空白部分
                zeros_y = np.zeros((step, self.feature_dim + 1))
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, self.feature_dim + 1))
                padded_data_label = np.concatenate((self.data_label[start:, :], zeros_y))
                split_data_label = padded_data_label
            else:
                split_data_label = self.data_label[start:end, :]
            # no part
            feature_data = split_data_label[:, :self.feature_dim] 
            label_data = np.array(split_data_label[:, -1]).squeeze()
            # 保存
            self.data.append(feature_data)
            self.labels.append(label_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data,label

if __name__ == '__main__':

    '''train_dataset = CustomDataset(Mpii_dir, modalities,modalities_dim)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1,prefetch_factor=4, pin_memory=True)
    print(len(trainloader))
    for i, (data,label) in enumerate(trainloader):
        print(data.shape)
        print(label.shape)
        break'''
