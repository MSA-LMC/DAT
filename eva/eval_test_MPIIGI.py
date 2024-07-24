
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
# from src.mymodel_cross_partnet import mymodel  # 假设你的模型类定义在这个路径下
import pandas as pd
# from src.model import CrossenhancedCEAM
# from src.mymodel_MPIIGI import mymodel
from src.mymodel_cross_origin import mymodel
from dataset_nopart_test import CustomDataset


'''
加载模型，对验证集或者测试集每个文件分别进行测试，然后将最终标签写入相关文件中。
'''



feature_len = 2477
#2477/1453

# 自定义Dataset类，用于被测数据集中数据的加载：
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
    def __init__(self, data_dir, modalities, role):

        self.data_dir = data_dir
        self.modalities = modalities  #模态名称列表
        self.data = []  #数据
        self.role = role
        self.data_label = np.empty((0, feature_len)) #用来存储这一对novice和expert所有的特征和标签，注意每次清零
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
                zeros_y = np.zeros((step, feature_len))
                #4956/2908
                padded_data_label = np.concatenate((zeros_y, self.data_label[:end, :]))
                split_data_label = padded_data_label
            elif end > data_label_len:  # 在元数据后面补充空白部分
                pad_length = end - data_label_len
                zeros_y = np.zeros((pad_length, feature_len))
                #4956/2908
                padded_data_label = np.concatenate((self.data_label[start:, :], zeros_y))
                split_data_label = padded_data_label
            else:
                split_data_label = self.data_label[start:end, :]


            #将这些data和label进行排序组合，变成（data, partner_data）的形式;

            self.data.append(split_data_label)  #数据

    def process_data(self):

        # 使用 tqdm 包装迭代器，显示进度条
        # 遍历anno_dict里面的成对的novice和expert文件，然后读取数据和标签，最终每个对输出一个data_label,进行切片
        entry = self.data_dir
        self.data_label = np.empty((0, feature_len)) #用来存储这一对novice和expert所有的特征和标签
        #4956/2908
        role_features = {}

        lengths = [] #把novice和expert所有的长度都记录下来

        base_path = entry

        # 对于每个novice和expert label 文件，找到其对应的特征文件，并且读取其特征,分别保存在novice_features和expert_features：
        for modality in self.modalities:
            role_modality_file = os.path.join(base_path, self.role + modality)
            role_stream_f = Stream().load(role_modality_file)
            role_features[modality] = role_stream_f.data
            lengths.append(role_stream_f.data.shape[0])

        #至此所有长度都得到了，取最小值，保证novice和expert所有的特征和标签数量全部对齐：
        num_samples = min(lengths)
        self.data_label = np.zeros((num_samples, feature_len))

        for i in range(num_samples):
            role_sample = np.concatenate([np.nan_to_num(role_features[modality][i]) for modality in self.modalities])
            self.data_label[i] = role_sample

        self.split_data_label()

        # print(len(self.data))

        # print(np.array(self.data).shape)
        # print(np.array(self.partner_data).shape)
        # print(np.array(self.labels).shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx])

#自定义数据标签的读取：
def read_label(label_dir, role):

    #按序遍历label文件夹，然后读取所有文件：
    if role == 'subjectPos1':
        file = os.path.join(label_dir, "subjectPos1.engagement.annotation.csv")
    elif role == "subjectPos2":
        file = os.path.join(label_dir, "subjectPos2.engagement.annotation.csv")
    elif role == "subjectPos3":
        file = os.path.join(label_dir, "subjectPos3.engagement.annotation.csv")
    else:
        file = os.path.join(label_dir, "subjectPos4.engagement.annotation.csv")


    # 读取 label 文件中的内容:
    label_values = read_label_file(file)

    return label_values


#定义函数进行label文件的数据读取：
def read_label_file(file_name):

    with open(file_name, "r") as f:
        label = np.genfromtxt(f, delimiter="\n", dtype=str)

    #预先申请内存进行label值的保存：
    #如果一个文件中的label不足32的倍数，就将其补全,保证与训练时一致：
    label_len = len(label)
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

#加载模型参数：
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # print(checkpoint.keys())
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)

    return model

# 定义归一化数据集类
class NormalizedDataset(Dataset):
    def __init__(self, dataset, global_min, global_max):
        self.dataset = dataset
        self.global_min = global_min
        self.global_max = global_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data = sample
        normalized_data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
        return normalized_data

# 对数据进行归一化：
def compute_min_max(dataloader):
    global_min = None
    global_max = None

    for batch in dataloader:
        data = batch  # 假设数据在第一个位置

        # 先沿第一个维度计算最小值和最大值
        batch_min_1 = torch.min(data, dim=0)[0]
        batch_max_1 = torch.max(data, dim=0)[0]

        # 再沿第二个维度计算最小值和最大值
        batch_min = torch.min(batch_min_1, dim=0)[0]
        batch_max = torch.max(batch_max_1, dim=0)[0]

        if global_min is None:
            global_min = batch_min
            global_max = batch_max
        else:
            global_min = torch.min(global_min, batch_min)
            global_max = torch.max(global_max, batch_max)

    return global_min.numpy(), global_max.numpy()

# 定义评估函数
def evaluate(model, device, label_dir, role, global_min, global_max):
    ypred = [] #定义模型预测值


    # 首先加载被测数据集的数据：
    val_dataset = CustomDataset_data(label_dir, modalities, role)
    val_dataset = NormalizedDataset(val_dataset, global_min, global_max)

    val_length = len(val_dataset)

    # 然后加载被测数据集的标签：
    # label_values = read_label(label_dir, role)
    # print('val_label: ', label_values.shape)

    eval_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, prefetch_factor=4,
                             pin_memory=True)

    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Preding: "):
            # print(data.shape)
            inputs = data
            inputs = inputs.float().to(device)
            outputs = model(inputs)


            ypred.append(outputs.cpu().numpy())
    ypred = np.concatenate(ypred, axis=0)  # 按行进行拼接
    reshaped_ypred = ypred.reshape(ypred.shape[0], ypred.shape[1])

    predict = np.zeros(val_length * 32 + 32)
    for j in range(reshaped_ypred.shape[0]):
        start = j * 32
        end = start + 32
        predict[start:end] += reshaped_ypred[j][32:32 * 2]

    pred_values = predict[:val_length * 32]

    for i in range(3,len(pred_values)-3):
        pred_values[i] = (pred_values[i-3]+pred_values[i-2]+pred_values[i-1]+pred_values[i]+pred_values[i+1]+pred_values[i+2]+pred_values[i+3])/7

    # pred_values = np.around(pred_values*24)/24


    # print(pred_values.shape)
    # print(label_values.shape)
    # #
    # label_values = label_values[:len(pred_values)]
    # #
    # ccc = concordance_correlation_coefficient(label_values, pred_values)
    # print(f"Concordance Correlation Coefficient: {ccc:.4f}")

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

    modalities_dim = [88, 1024, 512, 714, 139]

    # val_dataset = CustomDataset("/home/u2023110082/datasets/mpiigroupinteraction/", modalities, modalities_dim)
    all_dataset = CustomDataset("/home/u2023110082/NewDataset/MPIIGI/all/", modalities, modalities_dim)
    # val_dataset = CustomDataset("/home/u2023110082/NewDataset/MPIIGI/val/", modalities, modalities_dim)

    print("val_dataset loaded!")

    all_loader = DataLoader(all_dataset, batch_size=256, shuffle=True, num_workers=8, prefetch_factor=4,
                              pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=8, prefetch_factor=4,
    #                         pin_memory=True)

    global_min_all, global_max_all = compute_min_max(all_loader)
    # global_min_val, global_max_val = compute_min_max(val_loader)


    # 建立初始化模型：
    model_1 = mymodel()
    # model_2 = mymodel()
    # model_3 = mymodel()
    # model_4 = mymodel()
    # model_5 = mymodel()

    # 加载模型参数
    model_1 = nn.DataParallel(model_1)
    # model_2 = nn.DataParallel(model_2)
    # model_3 = nn.DataParallel(model_3)
    # model_4 = nn.DataParallel(model_4)
    # model_5 = nn.DataParallel(model_5)

    model_1 = model_1.to(device)
    # model_2 = model_2.to(device)
    # model_3 = model_3.to(device)
    # model_4 = model_4.to(device)
    # model_5 = model_5.to(device)




    # checkpoint_path_1 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_random_40.pt'
    # checkpoint_path_2 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_random_41.pt'
    # checkpoint_path_3 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_random_42.pt'
    # checkpoint_path_4 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_random_43.pt'
    checkpoint_path_1 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_random_44.pt'
    # checkpoint_path_1 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/M_best_score_random_40_all_normalized_cccloss.pt'

    # checkpoint_path_2 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/multimodal_MSEloss.pt'
    # checkpoint_path_2 = f'/home/u2023110082/yuyangchen/MEE24/NO1_lastyear/Dialogue-Cross-Enhanced-CEAM-main/model_weight/MPIIGI/mymodel_88_714_best_score_random_41.pt'

    model_1 = load_model(model_1, checkpoint_path_1)
    # model_2 = load_model(model_2, checkpoint_path_2)
    # model_3 = load_model(model_3, checkpoint_path_3)
    # model_4 = load_model(model_4, checkpoint_path_4)
    # model_5 = load_model(model_5, checkpoint_path_5)

    # 设置评估模式
    model_1.eval()
    # model_2.eval()
    # model_3.eval()
    # model_4.eval()
    # model_5.eval()

    anno = couple_label_path("/home/u2023110082/datasets/mpiigroupinteraction")
    # anno = couple_label_path("/home/u2023110082/NewDataset/MPIIGI/val")

    print(anno)

    pred_all = np.empty((0,))
    label_all = np.empty((0,))

    for i in anno:
        role_list = ["subjectPos1", "subjectPos2", "subjectPos3", "subjectPos4"]
        if i =="/home/u2023110082/datasets/mpiigroupinteraction/001":
            role_list = ["subjectPos1", "subjectPos2", "subjectPos3"]
        if i =="/home/u2023110082/NewDataset/MPIIGI/val/028":
            role_list = ["subjectPos2", "subjectPos3", "subjectPos4"]
        for j in role_list:

            # pred_1 = evaluate(model_1, device, i, j, global_min_all, global_max_all)
            # pred_2 = evaluate(model_2, device, i, j, global_min_all, global_max_all)
            # pred_3 = evaluate(model_3, device, i, j, global_min_all, global_max_all)
            # pred_4 = evaluate(model_4, device, i, j, global_min_all, global_max_all)
            pred_1 = evaluate(model_1, device, i, j, global_min_all, global_max_all)


            # 评估模型

            # pred_average = np.zeros((len(pred_1), ))
            #
            # for z in range(len(pred_1)):
            #     pred_average[z] = (pred_5[z]+pred_4[z]+pred_3[z]+pred_2[z]+pred_1[z])/5
            #     pred_average[z] = ( pred_2[z] + pred_1[z]) / 2

            # 然后加载被测数据集的标签：
            # label_values = read_label(i, j)
            #
            # #
            # pred_all = np.concatenate((pred_all, pred_1))
            # label_all = np.concatenate((label_all, label_values))
            # #
            # ccc = concordance_correlation_coefficient(label_values, pred_1)
            # print("pred_1: ", ccc)

            df = pd.DataFrame(pred_1)
            df.to_csv(os.path.join("/home/u2023110082/prediction/mpiigroupinteraction",os.path.basename(i), j+".engagement.annotation.csv"), header=False, index=False)
            print("saved!!!")
    # #
    # ccc = concordance_correlation_coefficient(label_all, pred_all)
    # print ("pred_all: ", ccc)












