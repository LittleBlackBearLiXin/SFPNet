import scipy.io as sio
from sklearn import preprocessing
import random
import os
import numpy as np





def load_dataset(flag):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if flag == 1:
        data_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_gt.mat')
        data = sio.loadmat(data_path)['indian_pines_corrected']
        gt = sio.loadmat(gt_path)['indian_pines_gt']
        class_count = 16
        dataset_name = "indian_"
    elif flag == 2:
        data_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU_gt.mat')
        data = sio.loadmat(data_path)['paviaU']
        gt = sio.loadmat(gt_path)['paviaU_gt']
        class_count = 9
        dataset_name = "paviaU_"
    elif flag == 3:
        data_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_gt.mat')
        data = sio.loadmat(data_path)['salinas_corrected']
        gt = sio.loadmat(gt_path)['salinas_gt']
        class_count = 16
        dataset_name = "salinas_"

    elif flag == 4:#https://github.com/GatorSense/MUUFLGulfport/blob/master/MUUFLGulfportSceneLabels/README.md
        data_path = os.path.join(parent_dir, '../HyperImage_data/MUUFL/muufl_gulfport_campus_1_hsi_220_label.mat')
        mat_data = sio.loadmat(data_path)
        hsi_struct = mat_data['hsi'][0, 0]  # 获取结构体内容
        data = hsi_struct['Data']  # 这应该是一个 (325, 220, 64) 的数据立方体
        scene_labels = hsi_struct['sceneLabels'][0, 0]  # sceneLabels也是一个结构体
        gt = scene_labels['labels']  # 这应该是一个 (325, 220) 的标签矩阵
        data = data.astype(np.float32)
        gt = gt.astype(np.int32)
        gt[gt == -1] = 0
        class_count = np.max(gt)
        dataset_name = "MUUFL"
    elif flag == 5:
        data_path = os.path.join(parent_dir, '../HyperImage_data/WHU-Hi-HongHu/WHU_Hi_HongHu.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat')
        data = sio.loadmat(data_path)['WHU_Hi_HongHu']
        gt = sio.loadmat(gt_path)['WHU_Hi_HongHu_gt']
        class_count = np.max(gt)
        dataset_name = "WHU_Hi_HongHu"
    elif flag == 6:
        data_path = os.path.join(parent_dir, '../HyperImage_data/HoustonU/Houston.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/HoustonU/Houston_gt.mat')
        data = sio.loadmat(data_path)['Houston']
        gt = sio.loadmat(gt_path)['Houston_gt']
        class_count =  np.max(gt)
        dataset_name = "Houston2013"
    else:
        raise ValueError("Invalid FLAG value")
    data = standardize_data(data)
    print(dataset_name, "Information:", data.shape, class_count, np.sum(gt > 0))

    return data, gt, class_count, dataset_name


def standardize_data(data):
    height, width, bands = data.shape
    reshaped = data.reshape(-1, bands)
    scaler = preprocessing.StandardScaler()
    normalized = scaler.fit_transform(reshaped).reshape(height, width, bands)
    return normalized



def ours_split(gt, seed, class_count):
    train_samples_per_class = 5
    val_samples_per_class = 5
    height, width = gt.shape
    total = height * width
    gt_reshape = gt.reshape(-1)
    random.seed(seed)


    train_idx = []
    for c in range(1, class_count + 1):
        idx = np.where(gt_reshape == c)[0]
        if len(idx) == 0:
            continue
        num = min(train_samples_per_class, len(idx) // 2)
        sampled = random.sample(list(idx), num)
        train_idx.extend(sampled)

    train_idx = set(train_idx)
    all_idx = set(np.where(gt_reshape > 0)[0])
    test_idx = all_idx - train_idx


    val_idx = set()
    for c in range(1, class_count + 1):
        class_test_idx = [i for i in test_idx if gt_reshape[i] == c]
        if len(class_test_idx) >= val_samples_per_class:
            sampled_val = random.sample(class_test_idx, val_samples_per_class)
            val_idx.update(sampled_val)

    test_idx -= val_idx


    def build_mask(index_set):
        mask = np.zeros(total)
        for i in index_set:
            mask[i] = gt_reshape[i]
        return mask.reshape(height, width)

    train_gt = build_mask(train_idx)
    val_gt = build_mask(val_idx)
    test_gt = build_mask(test_idx)


    def to_one_hot(gt_img):
        onehot = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_img.reshape(-1)
        for i in range(total):
            if flat[i] > 0:
                onehot[i, int(flat[i]) - 1] = 1
        return onehot

    return train_gt, val_gt, test_gt, to_one_hot(train_gt), to_one_hot(val_gt), to_one_hot(test_gt)



def split_data(gt, seed, class_count):

    train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot = ours_split(gt, seed, class_count)

    return train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot
