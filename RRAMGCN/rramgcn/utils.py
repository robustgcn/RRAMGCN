import numpy as np
import scipy.sparse as sp
import torch
import random
from deeprobust.graph.data import Dataset
from sklearn.model_selection import train_test_split

def preprocess_graph(adj, I=True):
    if I:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = adj_.sum(1).A1
    rowsum[rowsum == 0] = 1e-10  # 避免度为0时除以0
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(dataset='cora'):

    data = Dataset(root='data/', name=dataset)
    print("数据集是否有自带train_mask？", hasattr(data, 'train_mask') and data.train_mask is not None)  # 新增打印
    adj, features, labels = data.adj, data.features, data.labels

    # 确保邻接矩阵无向 + 无权
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj.setdiag(0)
    adj = adj.astype("float32")
    adj.eliminate_zeros()

    # 特征 -> 稠密张量
    features = torch.FloatTensor(features.toarray()).float()
    labels = torch.LongTensor(labels)

    if torch.cuda.is_available():
        features = features.cuda()
        labels = labels.cuda()

    # 划分数据集
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        split_train = np.where(data.train_mask)[0]
        split_val = np.where(data.val_mask)[0]
        split_unlabeled = np.where(data.test_mask)[0]
    else:
        idx = np.arange(len(labels))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            idx, labels.cpu(), train_size=0.1, stratify=labels.cpu(), random_state=42
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
        )
        split_train = idx_train
        split_val = idx_val
        split_unlabeled = idx_test

    return labels, features, split_train, split_val, split_unlabeled, adj

def set_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def csr_to_torch_sparse_tensor(csr_matrix, device=None, dtype=torch.float32):

    if not sp.issparse(csr_matrix):
        csr_matrix = sp.csr_matrix(csr_matrix)
    coo = csr_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float32)
    i = torch.LongTensor(indices)
    v = torch.from_numpy(values)
    shape = torch.Size(coo.shape)
    t = torch.sparse_coo_tensor(i, v, shape, dtype=dtype)
    t = t.coalesce()
    if device is not None:
        t = t.to(device)
    return t

def torch_sparse_to_dense(sp_tensor):

    if sp_tensor.is_sparse:
        return sp_tensor.coalesce().to_dense()
    else:
        return sp_tensor

def dense_to_csr(np_arr):
    import scipy.sparse as sp
    return sp.csr_matrix(np_arr)

def normalize_adj_sym(adj_csr, add_self_loops=True):

    if not sp.issparse(adj_csr):
        adj_csr = sp.csr_matrix(adj_csr)
    if add_self_loops:
        adj_csr = adj_csr + sp.eye(adj_csr.shape[0], dtype=adj_csr.dtype)
    adj_csr = adj_csr.tocoo()
    row_sum = np.array(adj_csr.sum(1)).flatten()
    # avoid div by zero
    row_sum[row_sum == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(row_sum)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized = D_inv_sqrt.dot(adj_csr).dot(D_inv_sqrt).tocsr()
    return normalized.astype(np.float32)

def eliminate_zeros_sparse(csr):
    csr.eliminate_zeros()
    return csr

def ensure_sparse_csr(adj):
    """确保邻接矩阵为CSR格式"""
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj_np = adj.coalesce().cpu().numpy()
            return sp.csr_matrix((adj_np.values(), adj_np.indices()), shape=adj_np.shape)
        else:
            return sp.csr_matrix(adj.cpu().numpy())
    elif sp.issparse(adj):
        return adj.tocsr()
    else:

        return sp.csr_matrix(adj)