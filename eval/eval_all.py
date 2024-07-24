import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from ssi_stream_utils import Stream
from src.metric import concordance_correlation_coefficient
# from src.mymodel_L_cross_partnet import mymodel  # 假设你的模型类定义在这个路径下
# from src.L_no1024 import mymodel
from src.mymodel_cross_partnet import mymodel

# from src.model import CrossenhancedCEAM

feature_len = 2477


# 2477/1453

# 自定义Dataset类，用于被测数据集中数据的加载：
# 存储所有成对标签的路径，返回anno字典：
def couple_label_path(data_dir):
    anno = []
    # 获取所有目录和文件路径并在最终字典里面排序,这样能够保证所有的数据在测试时与最终标签能够对齐：
    for path, sub_dirs, files in os.walk(data_dir):
        if files:
            anno.append(path)  # 这里的path是特征文件的文件夹路径如：'/dev/shm/yyc_data/Data/Noxi/val/007'
    anno.sort()  # 对文件夹进行排序，这样可以保证最后的标签顺序可控

    # print(anno)

    return anno


class CustomDataset_data(Dataset):
    def __init__(self, data_dir, modalities):

        self.data_dir = data_dir
        self.modalities = modalities  # 模态名称列表
        self.data = []  # 数据
        self.partner_data = []  # 同伴数据
        self.data_label = np.empty((0, 2 * feature_len))  # 用来存储这一对novice和expert所有的特征和标签，注意每次清零
        self.process_data()  # 数据加载的方法

    def split_data_label(self, role):  # 很呆的一个注释：第一个参数需要是self
        step = 32  # 滑动窗口的步长是32
        length = 96  # 滑动窗口的大小是96 [extended_len:core_len:extended_len]

        data_label_len = self.data_label.shape[0]

        # 确定按core_len切分之后的sample数量,core_len和step相等：
        if (data_label_len) % step != 0:
            n_samples = data_label_len // step + 1
        else:
            n_samples = data_label_len // step

        # 对data_label进行切割：
        for i in range(n_samples):
            start = i * step - step
            end = start + length
            if start < 0:  # 在原来数据上加了前面的空白部分
                zeros_y = np.zeros((step, 2 * feature_len))
                # 4956/2908
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, 2 * feature_len))
                # 4956/2908
                padded_data_label = np.concatenate((self.data_label[start:, :], zeros_y))
                split_data_label = padded_data_label
            else:
                split_data_label = self.data_label[start:end, :]

            # 对split_data_label进行切片，分别得到novice_data, expert_data, novice_label, expert_label：
            novice_data = split_data_label[:, :feature_len]
            # 2477/1453
            expert_data = split_data_label[:, feature_len:feature_len * 2]
            # 2477，4954/1453，2906

            # 将这些data和label进行排序组合，变成（data, partner_data）的形式;

            if role == 'novice':
                # 第一个样本（新手样本）：
                self.data.append(novice_data)  # 数据
                self.partner_data.append(expert_data)  # 同伴数据
            else:
                # 第二个样本(专家样本）：
                self.data.append(expert_data)  # 数据
                self.partner_data.append(novice_data)  # 同伴数据

    def process_data(self):

        for entry in tqdm(self.data_dir, desc="Loading Data: "):
            # 使用 tqdm 包装迭代器，显示进度条
            # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
            self.data_label = np.empty((0, 2 * feature_len))  # 用来存储这一对novice和expert所有的特征和标签
            # 4956/2908
            novice_features = {}
            expert_features = {}
            lengths = []  # 把novice和expert所有的长度都记录下来

            base_path = entry

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

            # 至此所有长度都得到了，取最小值，保证novice和expert所有的特征和标签数量全部对齐：
            num_samples = min(lengths)
            self.data_label = np.zeros((num_samples, 2 * feature_len))

            for i in range(num_samples):
                novice_sample = np.concatenate(
                    [np.nan_to_num(novice_features[modality][i]) for modality in self.modalities])
                expert_sample = np.concatenate(
                    [np.nan_to_num(expert_features[modality][i]) for modality in self.modalities])
                # 这里的总维度为 2477+2477+1+1 = 4956
                ne_data_label = np.concatenate((novice_sample, expert_sample))
                self.data_label[i] = ne_data_label

            self.split_data_label(role='expert')
            self.split_data_label(role='novice')

            # print(np.array(self.data).shape)
            # print(np.array(self.partner_data).shape)
            # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), np.array(self.partner_data[idx])


# 自定义数据标签的读取：
def read_label(label_dir):
    label_values = np.empty((0,))

    for label_path in tqdm(label_dir, desc="Loading Label: "):
        # 按序遍历label文件夹，然后读取所有文件：
        file_expert = os.path.join(label_path, "expert.engagement.annotation.csv")
        file_novice = os.path.join(label_path, "novice.engagement.annotation.csv")

        # 读取 label 文件中的内容:
        label_values = np.concatenate((label_values, read_label_file(file_expert)))
        label_values = np.concatenate((label_values, read_label_file(file_novice)))

    return label_values


# 定义函数进行label文件的数据读取：
def read_label_file(file_name):
    with open(file_name, "r") as f:
        label = np.genfromtxt(f, delimiter="\n", dtype=str)

    # 预先申请内存进行label值的保存：
    # 如果一个文件中的label不足32的倍数，就将其补全,保证与训练时一致：
    label_len = len(label)

    if len(label) % 32 != 0:
        label_len = (len(label) // 32 + 1) * 32

    label_values = np.zeros((label_len,))
    for i, label_value in enumerate(label):
        if label_value == '-nan(ind)':
            label_values[i] = float(0)
        else:
            label_values[i] = float(label_value)

    # 去除nan值：
    label_values = np.nan_to_num(label_values)
    # print(label_values.shape)

    return label_values


# 加载模型参数：
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # print(checkpoint.keys())
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)

    return model


# 定义评估函数
def evaluate(model, device, label_dir):
    ypred = []  # 定义模型预测值

    # 首先加载被测数据集的数据：
    val_dataset = CustomDataset_data(label_dir, modalities)
    val_length = len(val_dataset)

    # 然后加载被测数据集的标签：
    label_values = read_label(label_dir)
    # print('val_label: ', label_values.shape)

    eval_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, prefetch_factor=4,
                             pin_memory=True)

    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Preding: "):
            inputs, partnet_inputs = data
            inputs, partnet_inputs = inputs.float().to(device), partnet_inputs.float().to(device)
            outputs = model(inputs, partnet_inputs)

            ypred.append(outputs.cpu().numpy())
    ypred = np.concatenate(ypred, axis=0)  # 按行进行拼接
    reshaped_ypred = ypred.reshape(ypred.shape[0], ypred.shape[1])

    predict = np.zeros(val_length * 32 + 32)
    for j in range(reshaped_ypred.shape[0]):
        start = j * 32
        end = start + 32
        predict[start:end] += reshaped_ypred[j][32:32 * 2]

    pred_values = predict[:val_length * 32]

    print(pred_values.shape)
    print(label_values.shape)

    ccc = concordance_correlation_coefficient(label_values, pred_values)

    # print(f"Evaluation Loss: {running_loss / len(eval_loader):.4f}")
    print(f"Concordance Correlation Coefficient: {ccc:.4f}")
    return pred_values


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 特征列表
    modalities = [
        ".audio.egemapsv2.stream",
        ".audio.w2vbert2_embeddings.stream",  # 这个数据的采样率不一样，所以可以消融实验看看
        ".video.clip.stream",
        ".video.openface2.stream",
        ".video.openpose.stream",
    ]

    # 建立初始化模型：
    model_1 = mymodel()
    model_2 = mymodel()
    model_3 = mymodel()
    model_4 = mymodel()

    # 加载模型参数
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_3 = nn.DataParallel(model_3)
    model_4 = nn.DataParallel(model_4)

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_3 = model_3.to(device)
    model_4 = model_4.to(device)

    # checkpoint_path_1 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/best_loss_random_40.pt'
    # checkpoint_path_2 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/best_loss_random_41.pt'
    # checkpoint_path_3 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/best_loss_random_42.pt'
    # # checkpoint_path_4 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/best_loss_random_43.pt'
    checkpoint_path_1 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/M_partnet_32_best_score_random_40.pt'
    checkpoint_path_2 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/M_partnet_32_best_score_random_41.pt'
    checkpoint_path_3 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/M_partnet_32_best_score_random_42.pt'
    checkpoint_path_4 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/M_partnet_32_best_score_random_43.pt'

    # anno = couple_label_path("/dev/shm/yyc_data/Data/Noxi/val/")
    anno = couple_label_path("/home/u2023110082/NewDataset/Noxi/val/")

    model_1 = load_model(model_1, checkpoint_path_1)
    model_2 = load_model(model_2, checkpoint_path_2)
    model_3 = load_model(model_3, checkpoint_path_3)
    model_4 = load_model(model_4, checkpoint_path_4)

    # 设置评估模式
    model_1.eval()
    # model_1.train()
    model_2.eval()
    model_3.eval()
    model_4.eval()

    # 评估模型
    pred_1 = evaluate(model_1, device, anno)

    pred_2 = evaluate(model_2, device, anno)

    pred_3 = evaluate(model_3, device, anno)

    pred_4 = evaluate(model_4, device, anno)

    pred_average = np.zeros((len(pred_1),))

    for i in range(len(pred_1)):
        pred_average[i] = (pred_4[i]+pred_3[i]+pred_2[i]+pred_1[i])/4
        # pred_average[i] = (pred_1[i]+pred_2[i])/2

    # 然后加载被测数据集的标签：
    label_values = read_label(anno)
    ccc = concordance_correlation_coefficient(label_values, pred_average)
    # ccc = concordance_correlation_coefficient(label_values, pred_1)
    print("pred_average: ", ccc)





