# -*- coding: utf-8 -*-

# 使用绝对导入
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 使用 GPU 0 和 1  #这一行需要在import torch前面进行导入，这样才是指定卡


import numpy as np
import torch
torch.cuda.empty_cache() #清楚显卡中的显存占用

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import sys
import argparse
from src.metric import concordance_correlation_coefficient

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils import set_random_seed
from dataset_nopart  import  CustomDataset
# from dataset_MPIIGI_4 import OnlineDataset, CustomDataset
# from dataset_40to25 import OnlineDataset, CustomDataset
#from src.model import CrossenhancedCEAM
# from src.model_normalized import CrossenhancedCEAM
# from src.model_vit import CrossenhancedCEAM
# from src.noxi_model_no_partnet import CrossenhancedCEAM
# from src.mymodel import mymodel
# from src.mymodel_cross_origin import mymodel
# from src.mymodel_cross import mymodel
from mymodel_cross import mymodel
from src.loss import CenterMSELoss
from torch.utils.data import random_split
import copy

# 每个特征的维度：
modalities_dim = [
    88,
    1024,
    512,
    714,
    139
]

# EMA方法，防止模型过拟合：
def update_model_ema(model, model_ema, decay=0.99):
    net_g_params = dict(model.named_parameters())
    net_g_ema_params = dict(model_ema.named_parameters())

    for k in net_g_ema_params.keys():
        net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1 / np.sqrt(m.weight.size(1)))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 初始化权重函数
def xavier_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]




def parse_arguments():
    parser = argparse.ArgumentParser(description='CEAM Script Description')

    parser.add_argument('--save_dir', type=str, default='Cross_CEAM',
                        help='Save directory name')
    parser.add_argument('--position_embedding_type', type=str, default='fixed',
                        help='Position embedding type')
    parser.add_argument('--modality', type=str, default='multimodal',
                        help='Modality')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--N', type=int, default=1,
                        help='N')
    parser.add_argument('--M', type=int, default=1,
                        help='M')
    parser.add_argument('--K', type=int, default=2,
                        help='K')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weighted block skip connection')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Center Mse loss')
    parser.add_argument('--learning_rate', type=float, default=5e-5,  # 学习率跟随着batch_size的增大而成倍增大：
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension for attention')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--core_length', type=int, default=32,
                        help='Core length')
    parser.add_argument('--extended_length', type=int, default=32,
                        help='Extended length')
    parser.add_argument('--seed_value', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--to_continue', type=bool, default=False,
                        help='To Continue')
    parser.add_argument('--warmup_steps', type=int, default=400, help='Number of warmup steps')

    args = parser.parse_args()
    return args

class ExponentialSmoothedCCCLoss(nn.Module):
    def __init__(self, beta=0.5, alpha=0.9):
        super(ExponentialSmoothedCCCLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def exponential_smoothing(self, data):
        smoothed_data = torch.zeros_like(data)
        smoothed_data[0] = data[0]
        for t in range(1, data.size(0)):
            smoothed_data[t] = self.alpha * data[t] + (1 - self.alpha) * smoothed_data[t - 1]
        return smoothed_data

    def forward(self, y_true, y_pred):
        # 应用指数平滑
        y_true_smoothed = self.exponential_smoothing(y_true)
        y_pred_smoothed = self.exponential_smoothing(y_pred)
        
        # 计算均值
        mean_true = torch.mean(y_true_smoothed)
        mean_pred = torch.mean(y_pred_smoothed)

        # 计算方差
        var_true = torch.var(y_true_smoothed)
        var_pred = torch.var(y_pred_smoothed)

        # 计算标准差
        sd_true = torch.std(y_true_smoothed)
        sd_pred = torch.std(y_pred_smoothed)

        # 计算Pearson相关系数
        cor = torch.mean((y_true_smoothed - mean_true) * (y_pred_smoothed - mean_pred)) / (sd_true * sd_pred)

        # 计算CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator

        # 最大化ccc等价于最小化1-ccc
        loss = 1 - ccc
        return loss * self.beta

if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    modalities = [
        ".audio.egemapsv2.stream",          #特征维度88
        ".audio.w2vbert2_embeddings.stream",#特征维度1024
        ".video.clip.stream",               #特征维度512
        ".video.openface2.stream",          #特征维度714
        ".video.openpose.stream",           #特征维度139
    ]

    modalities_dim = [
        88,
        1024,
        512,
        714,
        139
    ]
    # 加载pre_dataset:
    mpiigi_train_dataset = CustomDataset("/data1/public_datasets/MEE_2024/MPIIGI/train_mini", modalities,modalities_dim)
    mpiigi_val_dataset = CustomDataset("/data1/public_datasets/MEE_2024/MPIIGI/val_mini", modalities,modalities_dim)

    # 对数据进行归一化：
    def compute_min_max(dataloader):
        global_min = None
        global_max = None

        for batch in dataloader:
            data = batch[0]  # 假设数据在第一个位置

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


    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1 / np.sqrt(m.weight.size(1)))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
            data = sample[0]
            normalized_data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
            return (normalized_data, *sample[1:])


    # 计算训练集的最小值和最大值
    train_dataloader = DataLoader(mpiigi_train_dataset, batch_size=128, shuffle=False)
    global_min, global_max = compute_min_max(train_dataloader)
    train_dataset = NormalizedDataset(mpiigi_train_dataset, global_min, global_max)

    val_dataloader = DataLoader(mpiigi_val_dataset, batch_size=256, shuffle=False)
    global_min, global_max = compute_min_max(val_dataloader)
    val_dataset = NormalizedDataset(mpiigi_val_dataset, global_min, global_max)


    val_length = len(val_dataset)

    trainloader = DataLoader(train_dataset, batch_size=128,num_workers=8,prefetch_factor=6,shuffle=True)

    # val_dataset = train_and_valDataset(args.data_path)
    valloader = DataLoader(val_dataset, batch_size=256,num_workers=8,prefetch_factor=6,shuffle=False)

    ff_dim = args.embed_dim * 4  # Hidden layer size in feed forward network inside transformer
    length = args.core_length + args.extended_length * 2
    max_position_embeddings = length

    #model = CrossenhancedCEAM(args.embed_dim, args.num_heads, ff_dim, args.N,args.M,args.K, args.dropout_rate,args.position_embedding_type,max_position_embeddings,args.alpha)
    model = mymodel()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据并行，将模型分别加载到不同的GPU,然后将批次分到不同的模型上进行训练：
    model = nn.DataParallel(model)
    model = model.to(device)

    model.apply(initialize_weights)

    model_ema = copy.deepcopy(model.module)
    model_ema = nn.DataParallel(model_ema)
    model_ema = model_ema.to(device)

    train_loss = []
    val_loss = []

    val_loss_flag = float('inf')
    keep_train = 0
    criterion = CenterMSELoss(beta=args.beta)
    #criterion = ExponentialSmoothedCCCLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00005)

    #添加warm_up的学习率：
    warmup_scheduler = WarmUpLR(optimizer, warmup_steps=args.warmup_steps, base_lr=args.learning_rate)

    for epoch in tqdm(range(args.epochs)):
        logging.debug(
            'Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        print('Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        # 设置模型为训练模式
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            try:
                inputs, labels = data
                inputs,labels = inputs.float().to(device),  labels.float().to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                # outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                # 更新ema_model:
                update_model_ema(model, model_ema, decay=0.99)

                running_loss += loss.item()

                # 更新warmup调度器
                if epoch * len(trainloader) + i < args.warmup_steps:
                    warmup_scheduler.step()


            except RuntimeError as e:
                if 'CUDA error' in str(e):
                    print("CUDA error encountered:", e)
                    torch.cuda.empty_cache()  # 清空缓存，尝试恢复
        train_loss.append(running_loss / len(trainloader))

        # 设置模型为评估模式
        model.eval()
        ypred = []
        labels_ = []
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(valloader, 0):
                # inputs, partnet_inputs,_, _,labels = data
                # inputs, partnet_inputs,labels = inputs.float().to(device), partnet_inputs.float().to(device), labels.float().to(device)

                inputs,  labels = data
                inputs,  labels = inputs.float().to(device),  labels.float().to(device)

                outputs = model_ema(inputs)
                # outputs = model_ema(inputs)

                # 这里输出的维度是[batch_size, 96]
                loss = criterion(outputs, labels)

                # 用于直接显示得分情况：
                ypred.append(outputs.cpu().numpy())
                labels_.append(labels.cpu().numpy())

                running_loss += loss.item()

            ypred = np.concatenate(ypred, axis=0)  # 按行进行拼接
            labels_ = np.concatenate(labels_, axis=0)
            reshaped_ypred = ypred.reshape(ypred.shape[0], ypred.shape[1])
            reshaped_label = labels_.reshape(labels_.shape[0], labels_.shape[1])

            predict = np.zeros(val_length * 32 + 32)
            for j in range(reshaped_ypred.shape[0]):
                start = j * 32
                end = start + 32
                predict[start:end] += reshaped_ypred[j][32:32 * 2]

            labels_all = np.zeros(val_length * 32 + 32)
            for j in range(reshaped_label.shape[0]):
                start = j * 32
                end = start + 32
                labels_all[start:end] += reshaped_label[j][32:32 * 2]

            allpred = predict[:val_length * 32]
            labels_all = labels_all[:val_length * 32]
            print(allpred.shape)
            print(labels_all.shape)
            print("multimodal : " + str(concordance_correlation_coefficient(allpred, np.asarray(labels_all))))

            val_loss.append(running_loss / len(valloader))
            # scheduler.step(val_loss[-1])
            scheduler.step()

        logging.debug(f"Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")


        if val_loss[-1] < val_loss_flag:
            keep_train = 0
            val_loss_flag = val_loss[-1]
            model_save_path = 'model/cs'
            os.makedirs(model_save_path, exist_ok=True)
            model_file_path = os.path.join(model_save_path, args.modality + '.pt')
            torch.save(model.state_dict(), model_file_path)
        else:
            keep_train += 1

        if keep_train > 60:
            break



