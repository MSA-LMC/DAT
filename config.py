# CSW Script Description
core_length = 32
extended_length = 32
seed_value = 42

#定义所有的模态和模态对应的维度：
modalities = [
    ".audio.egemapsv2.stream",
    ".audio.w2vbert2_embeddings.stream",
    # ".audio.transcript.npy",
    ".video.clip.stream",
    ".video.openface2.stream",
    ".video.openpose.stream",
    ]

modalities_dim = [
    88,
    1024,
    # 768,
    512,
    714,
    139
]

# 小组内的角色
Noxi_role_list = ['expert','novice']
Mpii_role_4_list = ['subjectPos1','subjectPos2','subjectPos3','subjectPos4']
Mpii_role_3_list = ['subjectPos2','subjectPos3','subjectPos4']

#原数据位置
Noxi_train_dir = "/data1/public_datasets/MEE_2024/Noxi/train_mini"
Noxi_val_dir = "/data1/public_datasets/MEE_2024/Noxi/val_mini"

Mpii_dir = "/data1/public_datasets/MEE_2024/MPIIGI/train"
Mpii_3_data_dir = "Mpii_3"
Mpii_4_data_dir = "Mpii_4"

phase = "train"