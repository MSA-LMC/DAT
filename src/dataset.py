from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader


#获取dataset,每个元组分别由（data, partnerdata,label)组成
class train_and_valDataset(Dataset):
    def __init__(self, data_path, id_list):
        self.data_list = []
        self.partner_data_list = []
        self.labels_list = []

        for id in id_list:
            cur_data_path = os.path.join(data_path,id)
            cur_data_path_list = os.listdir(cur_data_path)
            data_list_cur = []
            for i in range(len(cur_data_path_list)):
                if cur_data_path_list[i].split('_')[0] != 'label':
                    data_list_cur.append(cur_data_path_list[i])

            for i in range(len(data_list_cur)):
                name = data_list_cur[i]
                if 'expert' in name:
                    partner_name = name.replace('expert','novice')
                else:
                    partner_name = name.replace('novice','expert')

                #注意这里的partner_name并没有做区分，这个partner是expert还是novice:
                if os.path.exists(os.path.join(cur_data_path,partner_name)):
                    self.data_list.append(os.path.join(cur_data_path,name))
                    self.labels_list.append(os.path.join(cur_data_path,f"label_{name.split('_')[-2]}_{name.split('_')[-1].split('.')[0]}.npy"))
                    self.partner_data_list.append(os.path.join(cur_data_path,partner_name))
                else:
                    continue 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        #np.load是用来加载.npy数据的，npy适合小的数据可以直接保存numpy数组，而hdf5适合比较大的数据
        data = np.load(self.data_list[idx])
        partner_data = np.load(self.partner_data_list[idx])
        label = np.load(self.labels_list[idx])

        return data,partner_data, label

#用来加载test数据，但是很奇怪的是为什么会有flag来确定是否加载label呢，这个很奇怪偶
class testDataset(Dataset):
    def __init__(self,data_path, id, name,flag=True):
        self.data_list = []
        self.labels_list = []
        self.partner_data_list = []
        self.flag = flag

        cur_data_path = os.path.join(data_path,id)
        cur_data_path_list = os.listdir(cur_data_path)
        data_list_cur = []
        
        for i in range(len(cur_data_path_list)):
            if cur_data_path_list[i].split('_')[2] == name:
                data_list_cur.append(cur_data_path_list[i])

        for i in range(int(len(data_list_cur))):
            self.data_list.append(os.path.join(cur_data_path,f'frame_feature_{name}_{i}.npy'))
            if name == 'expert':
                partner_name = f'frame_feature_novice_{i}.npy'
            else:
                partner_name = f'frame_feature_expert_{i}.npy'
            self.partner_data_list.append(os.path.join(cur_data_path,partner_name))
            if self.flag:
                self.labels_list.append(os.path.join(cur_data_path,f'label_{name}_{i}.npy'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        partner_data = np.load(self.partner_data_list[idx])
        if self.flag:
            label = np.load(self.labels_list[idx])
            
            return data, partner_data,label
        else:
            return data, partner_data
    
