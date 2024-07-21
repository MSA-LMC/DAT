
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用 GPU 0 和 1
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import json
import pandas as pd
import argparse

from src.utils import set_random_seed
#from src.dataset import testDataset
#from src.model import CrossenhancedCEAM
from src.metric import concordance_correlation_coefficient
from mymodel_cross_partnet import mymodel
from dataset import OnlineDataset, CustomDataset

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
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=768, 
                        help='Embedding dimension for attention')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.0, 
                        help='Dropout rate')
    parser.add_argument('--core_length', type=int, default=32, 
                        help='Core length')
    parser.add_argument('--extended_length', type=int, default=32, 
                        help='Extended length')
    parser.add_argument('--seed_value', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    return args


 

if __name__ == "__main__":
    args = parse_arguments()
    #set_random_seed(args.seed_value)
    modalities = [
        ".audio.egemapsv2.stream",
        ".audio.w2vbert2_embeddings.stream",  # 这个数据的采样率不一样，所以可以消融实验看看
        ".video.clip.stream",
        ".video.openface2.stream",
        ".video.openpose.stream",
    ]
    Noxi_val_dataset = CustomDataset("/data/public_datasets/MEE_2024/Noxi/val", modalities, "Noxi_val")
    # Noxi_val_dataset=OnlineDataset(Noxi_val_dataset)

    valloader = DataLoader(Noxi_val_dataset, batch_size=256,num_workers=8,prefetch_factor=6,shuffle=False)
    print(len(Noxi_val_dataset))

    val_length = len(Noxi_val_dataset)


    ff_dim = args.embed_dim*4  # Hidden layer size in feed forward network inside transformer
    length = args.core_length + args.extended_length*2 
    max_position_embeddings = length


    eval_model = mymodel()
    state_dict = torch.load('model/noxi_random_40/multimodal.pt')

    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
       if k.startswith('module.'):
           new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
       else:
            new_state_dict[k] = v
    eval_model.load_state_dict(new_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_model = eval_model.to(device)
    eval_model.train()



    ypred = []
    labels_ = []
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):

            inputs, par, labels = data
            inputs, par, labels = inputs.float().to(device), par.float().to(device), labels.float().to(device)
            outputs = eval_model(inputs,par)

            # 用于直接显示得分情况：
            ypred.append(outputs.cpu().numpy())
            labels_.append(labels.cpu().numpy())

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
        for i in range(1, len(allpred)-1):
             allpred[i] = (allpred[i-1] +allpred[i] +allpred[i+1])/3
        print(allpred.shape)
        print(labels_all.shape)
        print("multimodal : " + str(concordance_correlation_coefficient(allpred, np.asarray(labels_all))))

        print("eval validation end~~~~~~")





     