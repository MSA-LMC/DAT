import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import pandas as pd
import argparse



from src.utils import set_random_seed
from src.model import CrossenhancedCEAM
from src.metric import concordance_correlation_coefficient
from ssi_stream_utils import Stream

# -*- coding: utf-8 -*-
import torch.nn as nn
from tqdm import tqdm
# from dataset import CustomDataset
from src.mymodel_cross_partnet import mymodel  # 假设你的模型类定义在这个路径下

feature_len = 2477



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

            #第一个样本：
            self.data.append(novice_data)  #数据
            self.partner_data.append(expert_data) #同伴数据
            self.labels.append(novice_label)  #数据标签
            #第二个样本：
            self.data.append(expert_data)  #数据
            self.partner_data.append(novice_data) #同伴数据
            self.labels.append(expert_label)  #数据标签



    def process_data(self):

        anno_dict = self.couple_label_path()

        # #进行检查一对输出有没有问题：
        anno = {}
        anno['novice_expert;007'] = anno_dict['novice_expert;007']

        # 使用 tqdm 包装迭代器，显示进度条
        # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
        for entry in tqdm(anno, desc="Loading " +self.phase+ " data"):
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

            self.split_data_label()

            # print(np.array(self.data).shape)
            # print(np.array(self.partner_data).shape)
            # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), np.array(self.partner_data[idx]), np.array(self.labels[idx])

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


class CustomDataset_data(Dataset):
    def __init__(self, data_dir, modalities, phase):

        self.data_dir = data_dir
        self.modalities = modalities  #模态名称列表
        self.data = []  #数据
        self.partner_data = [] #同伴数据
        self.data_label = np.empty((0, 2*feature_len)) #用来存储这一对novice和expert所有的特征和标签，注意每次清零
        self.phase = phase
        self.process_data()  #数据加载的方法



    def split_data_label(self): #很呆的一个注释：第一个参数需要是self
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
                zeros_y = np.zeros((step, 2*feature_len))
                #4956/2908
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, 2*feature_len))
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

            #将这些data和label进行排序组合，变成（data, partner_data）的形式;

            #第一个样本：
            self.data.append(novice_data)  #数据
            self.partner_data.append(expert_data) #同伴数据
            #第二个样本：
            self.data.append(expert_data)  #数据
            self.partner_data.append(novice_data) #同伴数据

    def process_data(self):

        entry = self.data_dir
        # 使用 tqdm 包装迭代器，显示进度条
        # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
        self.data_label = np.empty((0, 2*feature_len)) #用来存储这一对novice和expert所有的特征和标签
        #4956/2908
        novice_features = {}
        expert_features = {}
        lengths = [] #把novice和expert所有的长度都记录下来

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

        #至此所有长度都得到了，取最小值，保证novice和expert所有的特征和标签数量全部对齐：
        num_samples = min(lengths)
        self.data_label = np.zeros((num_samples, 2*feature_len))

        for i in range(num_samples):
            novice_sample = np.concatenate([np.nan_to_num(novice_features[modality][i]) for modality in self.modalities])
            expert_sample = np.concatenate([np.nan_to_num(expert_features[modality][i]) for modality in self.modalities])
            #这里的总维度为 2477+2477+1+1 = 4956
            ne_data_label = np.concatenate((novice_sample, expert_sample))
            self.data_label[i] = ne_data_label

        self.split_data_label()

        print(len(self.data))

        # print(np.array(self.data).shape)
        # print(np.array(self.partner_data).shape)
        # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]), np.array(self.partner_data[idx])

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model

# 特征列表
modalities = [
    ".audio.egemapsv2.stream",
    ".audio.w2vbert2_embeddings.stream",  # 这个数据的采样率不一样，所以可以消融实验看看
    ".video.clip.stream",
    ".video.openface2.stream",
    ".video.openpose.stream",
]

# 初始化模型并加载预训练权重
model = mymodel()

# 数据并行，将模型分别加载到不同的GPU,然后将批次分到不同的模型上进行训练：
model = nn.DataParallel(model)
model = model.to(device)
checkpoint_path = '/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/Noxi/best_score_random_43.pt'
model = load_model(model, checkpoint_path)

# 设置评估模式
model.eval()


anno = couple_label_path("/dev/shm/yyc_data/Data/Noxi/val/")
print(anno[0])

# 加载评估数据集
eval_dataset_1 = CustomDataset_data(anno[0], modalities, "Noxi_val")
eval_dataset_2 = CustomDataset("/dev/shm/yyc_data/Data/Noxi/val/", modalities, "Noxi_val")

val_length_1 = len(eval_dataset_1)
val_length_2 = len(eval_dataset_2)
print(val_length_1)
print(val_length_2)

eval_loader_1 = DataLoader(eval_dataset_1, batch_size=256, shuffle=False, num_workers=16, prefetch_factor=8, pin_memory=True)
eval_loader_2 = DataLoader(eval_dataset_2, batch_size=256, shuffle=False, num_workers=16, prefetch_factor=8, pin_memory=True)



# 定义评估函数
def evaluate(model, eval_loader_1, eval_loader_2, device):
    criterion = nn.MSELoss()
    running_loss = 0.0
    ypred_1 = []
    ypred_2 = []
    labels_ = []

    with torch.no_grad():
        for data in eval_loader_1:
            inputs, partnet_inputs = data
            inputs, partnet_inputs = inputs.float().to(device), partnet_inputs.float().to(device)

            outputs = model(inputs, partnet_inputs)


            ypred_1.append(outputs.cpu().numpy())

        for data in eval_loader_2:
            inputs, partnet_inputs, labels = data
            inputs, partnet_inputs, labels = inputs.float().to(device), partnet_inputs.float().to(
                device), labels.float().to(device)

            outputs = model(inputs, partnet_inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            ypred_2.append(outputs.cpu().numpy())
            labels_.append(labels.cpu().numpy())

    ypred_1 = np.concatenate(ypred_1, axis=0)  # 按行进行拼接
    print(ypred_1.shape)
    ypred_2 = np.concatenate(ypred_2, axis=0)  # 按行进行拼接
    print(ypred_2.shape)
    # print(labels_[:3])
    labels_ = np.concatenate(labels_, axis=0)
    reshaped_ypred_1 = ypred_1.reshape(ypred_1.shape[0], ypred_1.shape[1])
    reshaped_ypred_2 = ypred_2.reshape(ypred_2.shape[0], ypred_2.shape[1])
    reshaped_label = labels_.reshape(labels_.shape[0], labels_.shape[1])
    print(reshaped_label[0:2, :])

    predict_1 = np.zeros(val_length_1 * 32 + 32)
    for j in range(reshaped_ypred_1.shape[0]):
        start = j * 32
        end = start + 32
        predict_1[start:end] += reshaped_ypred_1[j][32:32 * 2]

    predict_2 = np.zeros(val_length_2 * 32 + 32)
    for j in range(reshaped_ypred_2.shape[0]):
        start = j * 32
        end = start + 32
        predict_2[start:end] += reshaped_ypred_2[j][32:32 * 2]

    labels_all = np.zeros(val_length_1 * 32 + 32)
    for j in range(reshaped_label.shape[0]):
        start = j * 32
        end = start + 32
        labels_all[start:end] += reshaped_label[j][32:32 * 2]

    allpred_1 = predict_1[:val_length_1 * 32]
    allpred_2 = predict_2[:val_length_2 * 32]
    labels_all = labels_all[:val_length_1 * 32]

    print(allpred_1[:100])
    print(allpred_2[:100])
    print(labels_all[:100])
    print(allpred_1.shape)
    print(allpred_2.shape)
    print(labels_all.shape)

    # #对预测的数据进行后处理： 每三个或者每五个取一个均值：
    # pred = np.zeros((len(allpred),))
    # for i in range(len(allpred)):
    #     if i+1 < len(allpred) and i>0:
    #         pred[i] = (allpred[i-1]+allpred[i]+allpred[i+1])/3
    #     else:
    #         pred[i] = allp




    ccc_1 = concordance_correlation_coefficient(allpred_1, np.asarray(labels_all))
    ccc_2 = concordance_correlation_coefficient(allpred_2, np.asarray(labels_all))
    print(f"Evaluation Loss: {running_loss / len(eval_loader_2):.4f}")
    print(f"Concordance Correlation Coefficient_1: {ccc_1:.4f}")
    print(f"Concordance Correlation Coefficient_2: {ccc_2:.4f}")


# 评估模型
evaluate(model, eval_loader_1, eval_loader_2, device)



    # output_path = f'./output_test/{args.save_dir}'
    # with open(f"./data/test_class_dict.json", 'r') as file:
    #     test_class_data = json.load(file)
    #
    # all_num = 0
    # for key, value in test_class_data.items():
    #     all_num += value
    #
    # allpred = np.zeros(all_num)
    #
    #
    # start_idx = 0
    # for attr, count in test_class_data.items():
    #     name = attr.split(';')[0]
    #     id = attr.split(';')[1]
    #     val_dataset = testDataset(args.data_path,id,name,False)
    #     valloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=16,prefetch_factor=2, pin_memory=True)
    #
    #     ypred = []
    #     inputs_ = []
    #     with torch.no_grad():
    #         for i, data in enumerate(valloader, 0):
    #
    #             inputs,add_data = data
    #             inputs= inputs.float().to(device)
    #             add_data= add_data.float().to(device)
    #             outputs = eval_model(inputs,add_data)
    #             ypred.append(outputs.cpu().numpy())
    #
    #             inputs_.append(inputs)
    #
    #
    #         ypred = np.concatenate(ypred, axis=0)
    #         reshaped_data = ypred.reshape(ypred.shape[0],ypred.shape[1])
    #
    #         if (count - step) // step != 0:
    #             n_samples = (count - step) // step + 1
    #         else:n_samples = (count - step) // step
    #
    #         predict = np.zeros(n_samples*step+step)
    #         for j in range(reshaped_data.shape[0]):
    #             start = j * step
    #             end = start + step
    #             predict[start:end] += reshaped_data[j][step:step*2]
    #
    #         allpred[start_idx:start_idx+count] = predict[:count]
    #         output = predict[:count]
    #         start_idx+=count
    #         print(start_idx)
    #
    #         df = pd.DataFrame(output)
    #         if not os.path.exists(f'{output_path}/{id}'):
    #             os.makedirs(f'{output_path}/{id}')
    #
    #         df.to_csv(f'{output_path}/{id}/{name}.engagement.annotation.csv', header=False, index=False)


     