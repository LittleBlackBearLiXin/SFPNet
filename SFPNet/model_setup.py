import torch
import time
from layers.getmask import process_mask
from SLIC.our_SLIC import Segmenttttt
from layers.SPFNet import SPFNet_F,SPFNet_A
import scipy.sparse as sp
import numpy as np



def SPFNet_F_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG):
    print('useing MPNN')
    height, width, bands = data.shape
    ls = Segmenttttt(data, class_count - 1)
    tic0 = time.perf_counter()
    Q, A = ls.SLIC_Process(data, scale=superpixel_scale)
    toc0 = time.perf_counter()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-data_knowledge costs time: ", LDA_SLIC_Time)
    A = torch.from_numpy(A).to(device)
    if isinstance(Q, sp.spmatrix):
        Q_coo = Q.tocoo()
        indices = np.vstack((Q_coo.row, Q_coo.col))
        Q = torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(Q_coo.data),
            torch.Size(Q_coo.shape),
            device=device
        )
    else:
        Q = torch.from_numpy(Q).to(device)



    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)


    net_input = to_tensor(np.array(data, dtype=np.float32))
    ticous = time.perf_counter()
    mask = process_mask(net_input, device, FLAG=FLAG)
    tocous = time.perf_counter()
    print('ous get mask time:', (tocous - ticous),"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    net = SPFNet_F(height=height, width=width, changel=bands, class_count=class_count,
                    mask=mask, q=Q, a=A, hide=128)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            net)


def SPFNet_A_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG):
    height, width, bands = data.shape
    ls = Segmenttttt(data, class_count - 1)
    tic0 = time.perf_counter()
    Q, A = ls.SLIC_Process(data, scale=superpixel_scale)
    toc0 = time.perf_counter()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-data_knowledge costs time: ", LDA_SLIC_Time)
    A = torch.from_numpy(A).to(device)
    if isinstance(Q, sp.spmatrix):
        Q_coo = Q.tocoo()
        indices = np.vstack((Q_coo.row, Q_coo.col))
        Q = torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(Q_coo.data),
            torch.Size(Q_coo.shape),
            device=device
        )
    else:
        Q = torch.from_numpy(Q).to(device)

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))
    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)


    net_input = to_tensor(np.array(data, dtype=np.float32))

    ticous = time.perf_counter()
    mask = process_mask(net_input, device, FLAG=FLAG)
    tocous = time.perf_counter()
    print('ous get mask time:', (tocous - ticous),"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    net = SPFNet_A(height=height, width=width, changel=bands, class_count=class_count,
                    mask=mask, q=Q, a=A, hide=128)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            net)
