import numpy as np
from skimage.segmentation import slic
import torch
import os
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
import time

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

def SegmentsLabelProcess(labels):

    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels




from scipy import sparse
class OurSlic(object):
    def __init__(self, HSI, n_segments=1000, compactness=0.01, max_iter=10, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = HSI.shape  #
        data = np.reshape(HSI, [height * width, bands])
        self.data = np.reshape(data, [height, width, bands])

    def get_Q_and_S_and_Segments(self):#2,2,3,1,4,
        img = self.data
        (h, w, d) = img.shape
        tic = time.time()
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness,
                        max_num_iter=self.max_iter,convert2lab=False, sigma=self.sigma,
                        enforce_connectivity=True,min_size_factor=self.min_size_factor,
                        max_size_factor=self.max_size_factor, slic_zero=False)
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        toc = time.time()
        print("SLIC time cost:",toc-tic,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        segments_1 = np.reshape(segments, [-1])
        rows, cols = [], []
        for i in range(superpixel_count):
            idx = np.where(segments_1 == i)[0]
            rows.extend(idx)
            cols.extend([i] * len(idx))
        data = np.ones(len(rows), dtype=np.float32)
        Q = sparse.csr_matrix((data, (rows, cols)), shape=(w * h, superpixel_count))
        return Q


    def get_A(self):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                         continue
                    A[idx1, idx2] = A[idx2, idx1] = 1

        return A




class Segmenttttt(object):
    def __init__(self, data, n_component):
        self.data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape


    def SLIC_Process(self, img, scale=0):
        n_segments_init = self.height * self.width / scale
        myslic = OurSlic(img, n_segments=n_segments_init, sigma=1, min_size_factor=0.1,max_size_factor=10)#SLICcccc
        Q= myslic.get_Q_and_S_and_Segments()
        A = myslic.get_A()
        print("Number of superpixels:", A.shape[0])
        return Q,  A






def noramlize(A,aggr='add'):
    A = A + torch.eye(A.shape[0], A.shape[0]).to(device)
    if aggr == 'mean':
        return F.normalize(A,dim=1,p=1)
    else:
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        A = torch.mm(torch.mm(D_hat, A), D_hat)
        return A

def spareaA(adjacency,aggr):
    adjacency = noramlize(adjacency,aggr)
    total = (adjacency.shape[0])*(adjacency.shape[0])
    sparatio = 1 - (torch.sum(adjacency > 0)/total)
    if not adjacency.is_sparse:
        adjacency = adjacency.to_sparse()
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    size = adjacency.size()
    adjacency = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32, device=device)
    adjacency = adjacency.to_sparse_coo().coalesce()

    return adjacency, sparatio


