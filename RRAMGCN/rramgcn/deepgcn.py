import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from rramgcn.utils import csr_to_torch_sparse_tensor, normalize_adj_sym, set_device, eliminate_zeros_sparse

device = set_device()

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 tau=1.0, clamp_sim_min=False, attn_drop=0.0, eps=1e-8):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.tau = float(tau)
        self.clamp_sim_min = bool(clamp_sim_min)
        self.attn_drop = float(attn_drop)
        self.eps = float(eps)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):

        h = torch.matmul(x, self.weight)

        if getattr(adj, "is_sparse", False):
            adj_dense = adj.coalesce().to_dense().to(h.device).to(h.dtype)
        else:
            adj_dense = adj.to(h.device).to(h.dtype)

        h_norm = F.normalize(h, p=2, dim=1, eps=1e-8)
        sim = torch.matmul(h_norm, h_norm.t())

        if self.clamp_sim_min:
            sim = torch.clamp(sim, min=0.0)

        tau = self.tau if self.tau > 0 else 1.0
        sim_scaled = sim / tau

        adj_plus = adj_dense + self.eps
        attn_logits = sim_scaled + torch.log(adj_plus)

        attn = F.softmax(attn_logits, dim=1)

        if self.attn_drop > 0.0:
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)

        out = torch.matmul(attn, h)

        if self.bias is not None:
            out = out + self.bias

        return out


def adjPrecess(adj, adj_high, features, sim_threshold=0.1, k_top=7):

    if sp.issparse(features):
        fea_copy = features.toarray()
    else:
        fea_copy = np.array(features)
    fea_copy = fea_copy.astype(np.float32)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy).astype(np.float32)

    adj = adj.tocsr().copy()
    adj_high = adj_high.tocsr().copy()

    update_mask = sim_matrix < sim_threshold

    adj_coo = adj.tocoo()
    mask_indices = update_mask[adj_coo.row, adj_coo.col]
    adj_coo.data[mask_indices] = 0.0
    adj_coo.eliminate_zeros()
    updated_adj = adj_coo.tocsr()

    adj_coo_high = adj_high.tocoo()
    high_mask = update_mask[adj_coo_high.row, adj_coo_high.col]
    adj_coo_high.data[high_mask] = 1.0
    adj_coo_high.eliminate_zeros()
    updated_adj_high = adj_coo_high.tocsr()

    num_nodes = updated_adj.shape[0]
    k = min(k_top, num_nodes - 1)
    top_k = np.argpartition(-sim_matrix, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(num_nodes), k)
    cols = top_k.flatten()
    keep = rows != cols
    rows = rows[keep]
    cols = cols[keep]
    vals = np.ones(len(rows), dtype=np.float32)
    T_k = csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    updated_adj = (updated_adj + T_k).tocsr()
    updated_adj = (updated_adj + updated_adj.T).tocsr()
    updated_adj.data[updated_adj.data > 1] = 1.0
    updated_adj.eliminate_zeros()

    updated_adj_high = (updated_adj_high + updated_adj_high.T).tocsr()
    updated_adj_high.data[updated_adj_high.data > 1] = 1.0
    updated_adj_high.eliminate_zeros()
    return updated_adj, updated_adj_high, sim_matrix


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers=2, dropout=0.5,
                 attention_tau=1.0, clamp_sim_min=False, attn_drop=0.0):
        super(DeepGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers_low = nn.ModuleList()
        self.layers_high = nn.ModuleList()
        self.layers_mid = nn.ModuleList()
        self.gates = nn.ParameterList()
        in_dims = [nfeat] + [nhid] * (num_layers - 1)
        out_dims = [nhid] * (num_layers - 1) + [nclass]
        for l in range(num_layers):

            self.layers_low.append(GraphConv(in_dims[l], out_dims[l],
                                             tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop))
            self.layers_high.append(GraphConv(in_dims[l], out_dims[l],
                                              tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop))
            self.layers_mid.append(GraphConv(in_dims[l], out_dims[l],
                                             tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop))
            gate_param = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
            self.gates.append(gate_param)
        self.activation = nn.ReLU(inplace=True)
        self.to(device)

    def forward(self, x, data):
        adj_low = data.get('low_adj')
        adj_high = data.get('high_adj')
        adj_mid = data.get('mid_adj')
        h = x
        reps = [h]
        for l in range(self.num_layers):
            hl = self.layers_low[l](h, adj_low)
            hh = self.layers_high[l](h, adj_high)
            hm = self.layers_mid[l](h, adj_mid)
            if l < self.num_layers - 1:
                hl = self.activation(hl)
                hh = self.activation(hh)
                hm = self.activation(hm)
                hl = F.dropout(hl, p=self.dropout, training=self.training)
                hh = F.dropout(hh, p=self.dropout, training=self.training)
                hm = F.dropout(hm, p=self.dropout, training=self.training)
            weights = F.softmax(self.gates[l], dim=0)
            h = weights[0] * hl + weights[1] * hh + weights[2] * hm
            reps.append(h)
        out = F.log_softmax(h, dim=-1)
        return out, reps


class SmallGCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5,
                 attention_tau=1.0, clamp_sim_min=False, attn_drop=0.0):
        super(SmallGCN, self).__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim, tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop)
        self.gc2 = GraphConv(hidden_dim, out_dim, tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop)
        self.dropout = dropout

    def forward(self, x, adj):
        h = self.gc1(x, adj)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gc2(h, adj)
        return out, h


def simple_get_gcn_for_graphreshape(labels,
                                    features,
                                    adj_csr,
                                    split_train,
                                    split_val=None,
                                    split_unlabeled=None,
                                    epochs=50,
                                    h_dim=50,
                                    dropout=0.5,
                                    attention_tau=1.0,
                                    clamp_sim_min=False,
                                    attn_drop=0.0):
    in_dim = features.shape[1]
    num_classes = int(labels.max().item() + 1)

    model = SmallGCN(in_dim, h_dim, num_classes, dropout=dropout,
                     attention_tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)

    adj_norm = normalize_adj_sym(adj_csr, add_self_loops=True)
    adj_t = csr_to_torch_sparse_tensor(adj_norm, device=device)

    x = features.to(device)
    labels_t = labels.to(device)

    split_train = torch.LongTensor(split_train).to(device)
    if split_val is not None:
        split_val = torch.LongTensor(split_val).to(device)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, adj_t)
        loss_train = F.cross_entropy(logits[split_train], labels_t[split_train])
        loss_train.backward()
        optimizer.step()

        if split_val is not None and len(split_val) > 0:
            model.eval()
            with torch.no_grad():
                logits_val, _ = model(x, adj_t)
                loss_val = F.cross_entropy(logits_val[split_val], labels_t[split_val])
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, None