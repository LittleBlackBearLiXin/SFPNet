import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_seed(42)




def get_mask(net_input, device, num_blocks=4, token=0):

    height, width, _ = net_input.shape

    out = net_input.reshape([height * width, -1]).to(device)
    out = F.layer_norm(out, [out.size(-1)])
    N = height * width

    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    coordinates = torch.stack((rows.flatten(), cols.flatten()), dim=1)

    mask_src, mask_dst = [], []

    block_height = height // num_blocks
    block_width = width // num_blocks
    stride_height = block_height // 2
    stride_width = block_width // 2

    for i in range(0, height - block_height + 1, stride_height):
        for j in range(0, width - block_width + 1, stride_width):
            start_row = i
            end_row = min(i + block_height, height)
            start_col = j
            end_col = min(j + block_width, width)
            block_mask = (coordinates[:, 0] >= start_row) & (coordinates[:, 0] < end_row) & \
                         (coordinates[:, 1] >= start_col) & (coordinates[:, 1] < end_col)
            curr_indices = torch.where(block_mask)[0]

            if len(curr_indices) > 1:
                curr_out = out[curr_indices]
                curr_coords = coordinates[curr_indices]
                coord_sim = torch.exp(-(torch.cdist(curr_coords.float(), curr_coords.float(), p=2) ** 2)/ 4)
                total_k = min(token, len(curr_indices)//100)
                print(total_k,"*********") if total_k>token else None
                spatial_k = min(0, total_k)
                spatial_topk_val, spatial_topk_idx = torch.topk(coord_sim, k=spatial_k, dim=1)
                remaining_k = total_k - spatial_k
                if remaining_k > 0:
                    feat_sim = torch.relu(torch.mm(curr_out, curr_out.T))
                    com_sim = feat_sim * coord_sim
                    com_sim[torch.arange(len(com_sim))[:, None], spatial_topk_idx] = -1e9
                    remaining_topk_val, remaining_topk_idx = torch.topk(com_sim, k=remaining_k, dim=1)
                    all_topk_idx = torch.cat([spatial_topk_idx, remaining_topk_idx], dim=1)
                else:
                    all_topk_idx = spatial_topk_idx
                src = curr_indices.view(-1, 1).expand(-1, total_k)
                dst = curr_indices[all_topk_idx]
                mask_src.append(src.flatten())
                mask_dst.append(dst.flatten())

    if mask_src:
        sparse_mask = torch.sparse_coo_tensor(
            torch.stack([torch.cat(mask_src), torch.cat(mask_dst)]),
            torch.ones(sum(len(es) for es in mask_src), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).coalesce().to_sparse_csr()
    else:
        sparse_mask = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).to_sparse_csr()

    return sparse_mask


from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_add

def simfunction(mask):
    mask=mask.to_sparse_coo()
    mask_index = mask.indices()
    mask_attr = mask.values()

    mask_index, mask_attr = remove_self_loops(mask_index, mask_attr)
    mask_index, mask_attr = add_self_loops(mask_index, mask_attr, num_nodes=mask.size(0))

    src, dst = mask_index[0], mask_index[1]
    row_sum = scatter_add((mask_attr), src, dim_size=mask.shape[0])**(-0.5)
    col_sum = scatter_add((mask_attr), dst, dim_size=mask.shape[0])**(-0.5)
    att = (row_sum[src]) * mask_attr * (col_sum[dst])

    mask = torch.sparse_coo_tensor(
        mask_index,
        att,
        size=mask.size(),
        device=mask.device
    ).coalesce()

    return mask





def process_mask(net_input, device,FLAG=-1):
    row_num_subgraphs,row_top_k=hyp(FLAG)
    mask = get_mask(net_input, device, row_num_subgraphs, token=row_top_k)
    mask = simfunction(mask)
    total_elements = mask.size(0) * mask.size(1)
    non_zero_elements = mask._nnz()
    sparsity = 1.0 - (non_zero_elements / total_elements)
    print('pixel-level adjacency spare ratio:', sparsity)

    return mask










def hyp(FLAG):
    if FLAG == 1:
        num_subgraphs = 2
        token = 30
    elif FLAG == 2:
        num_subgraphs = 4#4
        token = 10#10
    elif FLAG == 3:
        num_subgraphs = 4#4
        token = 10#20
    elif FLAG == 4:
        num_subgraphs = 3#3
        token = 10#10
    elif FLAG == 5:
        num_subgraphs = 4
        token = 10
    elif FLAG == 6:
        num_subgraphs = 6
        token = 10
    else:
        num_subgraphs = 0
        token = 0

    return num_subgraphs, token
