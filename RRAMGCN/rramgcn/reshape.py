import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil

from rramgcn.utils import csr_to_torch_sparse_tensor, set_device, normalize_adj_sym

device = set_device()

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(Encoder, self).__init__()
        self.h1 = nn.Linear(x_dim, h_dim)
        self.act = nn.PReLU()

    def forward(self, x, adj=None, neighbor=True):
        if neighbor and adj is not None:
            if getattr(adj, "is_sparse", False):
                neigh = torch.sparse.mm(adj.coalesce(), x)
            else:
                neigh = torch.matmul(adj, x)
            h = self.h1(neigh)
        else:
            h = self.h1(x)
        h = self.act(h)
        return F.softmax(h, dim=1)


class GraphReshapeModel(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GraphReshapeModel, self).__init__()
        self.embedding = Encoder(x_dim, h_dim)
        self.to(device)

    def forward(self, x, adj):
        h_node = self.embedding(x, adj=None, neighbor=False)
        h_graph = self.embedding(x, adj=adj, neighbor=True)
        return h_node, h_graph


def train_graphreshape(model, features, adj_sparse, epochs=2000, optimizer=None,
                       n_sample=1, early_stop=400, device=device):
    best_loss = float('inf')
    best_state = None
    patience = early_stop
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=5e-4)
    x = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        p_node, p_graph = model(x, adj_sparse)

        cos_sim = torch.cosine_similarity(p_node, p_graph, dim=1).abs()
        a_val = cos_sim.mean()
        a = torch.min(a_val, torch.tensor(0.7, device=device))

        b_acc = 0.0
        N = x.size(0)
        for _ in range(max(1, n_sample)):
            idx = torch.randperm(N, device=device)
            b_acc += torch.max(torch.tensor(0.2, device=device),
                               torch.cosine_similarity(p_node[idx], p_graph, dim=1).abs().mean())
        b = b_acc / max(1, n_sample)

        loss = 1.0 - a + b
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience = early_stop
        else:
            patience -= 1
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class SmallGCN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim):
        super(SmallGCN, self).__init__()
        self.h1 = nn.Linear(x_dim, h_dim)
        self.h2 = nn.Linear(h_dim, y_dim)
        self.to(device)

    def forward(self, x, adj):
        if getattr(adj, "is_sparse", False):
            h = torch.sparse.mm(adj.coalesce(), x)
            h = F.relu(self.h1(h))
            emb = torch.sparse.mm(adj.coalesce(), h)
            y = self.h2(emb)
        else:
            pre_h = torch.matmul(adj, x)
            h = F.relu(self.h1(pre_h))
            emb = torch.matmul(adj, h)
            y = self.h2(emb)
        return y, emb


def simple_get_gcn_for_graphreshape(labels, features, adj_csr,
                                    split_train, split_val, split_unlabeled,
                                    epochs=50, h_dim=50,
                                    attention_tau=1.0, clamp_sim_min=False, attn_drop=0.0):
    x_dim = features.shape[-1]
    y_dim = int(labels.max().item() + 1) if torch.is_tensor(labels) else int(labels.max() + 1)
    try:
        from rramgcn.deepgcn import simple_get_gcn_for_graphreshape as rede_sg
        return rede_sg(labels, features, adj_csr, split_train, split_val, split_unlabeled,
                       epochs=epochs, h_dim=h_dim, dropout=attn_drop,
                       attention_tau=attention_tau, clamp_sim_min=clamp_sim_min, attn_drop=attn_drop)
    except Exception:
        model = SmallGCN(x_dim, h_dim, y_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)

        adj_norm = normalize_adj_sym(adj_csr, add_self_loops=True)
        adj_t = csr_to_torch_sparse_tensor(adj_norm, device=device)

        x = features.to(device)
        labels_t = labels.to(device) if torch.is_tensor(labels) else torch.LongTensor(labels).to(device)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x, adj_t)[0][split_train]
            loss = F.cross_entropy(logits, labels_t[split_train])
            loss.backward()
            optimizer.step()
        return model, None

def simple_finetune_gcn_for_graphreshape(model, adj_t, features, labels, split_train, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)
    x = features.to(device)
    labels_t = labels.to(device) if torch.is_tensor(labels) else torch.LongTensor(labels).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj_t)[0][split_train]
        loss = F.cross_entropy(logits, labels_t[split_train])
        loss.backward()
        optimizer.step()
    return model

def build_candidate_mask_from_sim(base_adj, sim_matrix, important_nodes, candidate_k=None, max_candidates=None):
    """
    Build a list of unique unordered candidate pairs (u,v) (u < v).
    If candidate_k is None, set dynamically based on average degree.
    """
    num_nodes = base_adj.shape[0]
    deg = base_adj.sum(axis=1).A1 if sp.issparse(base_adj) else np.array(base_adj.sum(axis=1)).flatten()
    avg_deg = float(np.mean(deg)) if deg.size > 0 else 1.0
    if candidate_k is None:
        candidate_k = min(15, max(5, int(avg_deg / 2)))
    k = min(candidate_k, num_nodes - 1)

    cand_set = set()
    for u in important_nodes:
        sims = sim_matrix[u]
        topk = np.argpartition(-sims, kth=k, axis=0)[:k]
        for v in topk:
            if v == u:
                continue
            a, b = (int(u), int(v))
            if a > b:
                a, b = b, a
            cand_set.add((a, b))
        # include existing neighbors
        nbrs = np.where(base_adj[u] > 0)[0] if isinstance(base_adj, np.ndarray) else base_adj[u].nonzero()[1].tolist()
        for v in nbrs:
            if v == u:
                continue
            a, b = (int(u), int(v))
            if a > b:
                a, b = b, a
            cand_set.add((a, b))

    if max_candidates is not None and len(cand_set) > max_candidates:
        cand_list = list(cand_set)[:max_candidates]
    else:
        cand_list = list(cand_set)
    return cand_list


def graphreshape(raw_adj, features, labels, split_train, split_val, split_unlabeled,
                 n_sample=1, lr=5e-4, weight_decay=5e-6, h_dim=50, threshold=0.7, max_finetune_iters=3,
                 importance_alpha=0.6, importance_beta=0.4, candidate_k=None, max_changes_per_iter=1,
                 recompute_every=10, max_candidates=5000):
    import scipy.sparse as sp_local
    if not sp_local.issparse(raw_adj):
        raw_adj = sp_local.csr_matrix(raw_adj)

    adj_norm = normalize_adj_sym(raw_adj, add_self_loops=True)
    adj_torch = csr_to_torch_sparse_tensor(adj_norm, device=device)

    x = features.to(device)
    model = GraphReshapeModel(x.size(-1), h_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model = train_graphreshape(model, features, adj_torch,
                               epochs=1000, optimizer=optimizer, n_sample=n_sample)

    with torch.no_grad():
        h_node, h_graph = model(features.to(device), adj_torch)
        score = torch.cosine_similarity(h_node, h_graph, dim=1).cpu().numpy()

    thr = threshold * 0.7 + (1 - threshold) * 0.2
    abnormal_node = np.where(score < thr)[0]
    abnormal_train = [int(i) for i in abnormal_node if int(i) in set(split_train)]

    raw_adj_mutable = raw_adj.tolil().astype(float).copy()
    for i in abnormal_train:
        raw_adj_mutable[i, :] = 0
        raw_adj_mutable[:, i] = 0
    raw_adj_mutable = raw_adj_mutable.tocsr()

    if len(abnormal_train) == 0:
        return raw_adj, []

    from rramgcn.deepgcn import simple_get_gcn_for_graphreshape as rede_sg
    GCN_model = rede_sg(labels, features, raw_adj_mutable,
                        split_train, split_val, split_unlabeled, epochs=50)[0]

    raw_adj_mutable = raw_adj_mutable.tolil()
    num_nodes = raw_adj_mutable.shape[0]
    # heuristic number of iterations
    num_iters = int(0.5 * raw_adj_mutable.sum() * max(1, len(abnormal_train)) / float(num_nodes))
    num_iters = max(10, min(num_iters, 5000))

    # compute base sim matrix once (features -> sim)
    from sklearn.metrics.pairwise import cosine_similarity
    feat_np = features.cpu().numpy() if hasattr(features, 'cpu') else np.array(features)
    sim_matrix = cosine_similarity(feat_np, feat_np).astype(np.float32)

    # build initial candidate list
    cand_pairs = build_candidate_mask_from_sim(raw_adj_mutable.toarray(), sim_matrix, abnormal_train,
                                              candidate_k=candidate_k, max_candidates=max_candidates)

    for it in tqdm(range(num_iters), desc='GraphReshape updates'):
        # periodically recompute candidate list
        if recompute_every is not None and recompute_every > 0 and (it % recompute_every == 0) and it > 0:
            cand_pairs = build_candidate_mask_from_sim(raw_adj_mutable.toarray(), sim_matrix, abnormal_train,
                                                      candidate_k=candidate_k, max_candidates=max_candidates)

        # normalize and convert to sparse tensor for fine-tuning
        adj_norm = normalize_adj_sym(raw_adj_mutable.tocsr(), add_self_loops=True)
        adj_t = csr_to_torch_sparse_tensor(adj_norm, device=device)
        GCN_model = simple_finetune_gcn_for_graphreshape(GCN_model, adj_t,
                                                         features, labels, split_train,
                                                         epochs=max_finetune_iters)
        GCN_model.eval()

        # prepare dense leaf adj for gradient computation
        adj_dense = adj_t.to_dense().to(device)
        adj_dense = adj_dense.clone().detach().requires_grad_(True)

        x = features.to(device)
        logits = GCN_model(x, adj_dense)[0]
        loss = F.cross_entropy(logits[split_train], labels[split_train].to(device))

        # backward for grad
        if adj_dense.grad is not None:
            adj_dense.grad.zero_()
        loss.backward()

        grad = None
        if adj_dense.grad is not None:
            grad = adj_dense.grad.detach().cpu().numpy()
        else:
            try:
                g = torch.autograd.grad(loss, adj_dense, retain_graph=False, create_graph=False)
                grad = g[0].detach().cpu().numpy() if (g is not None and g[0] is not None) else None
            except Exception:
                grad = None

        if grad is None:
            # can't proceed if we don't have grad
            # print warning and break
            print("Warning: grad is None in GraphReshape iter; breaking out.")
            break

        base = raw_adj_mutable.toarray()
        # original gradient adjustment in previous code
        grad = grad - 1 + 2 * base

        # compute structure importance matrices
        deg = base.sum(axis=1).astype(np.float32)
        deg_sum = deg[:, None] + deg[None, :]
        deg_sum_min = deg_sum.min()
        deg_sum_max = deg_sum.max()
        deg_sum_range = (deg_sum_max - deg_sum_min) if (deg_sum_max - deg_sum_min) > 0 else 1.0
        deg_sum_norm = (deg_sum - deg_sum_min) / deg_sum_range

        common = base.dot(base).astype(np.float32)
        deg_min = np.minimum(deg[:, None], deg[None, :])
        deg_min[deg_min == 0] = 1.0
        common_ratio = common / deg_min
        common_ratio = np.clip(common_ratio, 0.0, 1.0)

        importance = importance_alpha * deg_sum_norm + importance_beta * (1.0 - common_ratio)
        importance = np.maximum(importance, 0.0)

        # restrict to abnormal rows/cols
        mask_ab = np.zeros_like(grad, dtype=bool)
        mask_ab[np.ix_(abnormal_train, np.arange(num_nodes))] = True
        mask_ab[np.ix_(np.arange(num_nodes), abnormal_train)] = True

        # restrict further to candidates in cand_pairs
        candidate_mask = np.zeros_like(grad, dtype=bool)
        for (u, v) in cand_pairs:
            candidate_mask[u, v] = True
            candidate_mask[v, u] = True

        final_mask = mask_ab & candidate_mask

        # compute add/remove candidate scores
        importance_masked = np.where(final_mask, importance, 0.0)
        grad_masked = np.where(final_mask, grad, 0.0)

        add_candidates = (base <= 0.5) & (grad_masked < 0) & final_mask
        add_scores = np.where(add_candidates, (-grad_masked) * importance_masked, 0.0)

        remove_candidates = (base > 0.5) & (grad_masked > 0) & final_mask
        remove_scores = np.where(remove_candidates, grad_masked * importance_masked, 0.0)

        # merge candidates as unique unordered edges (u < v)
        edge_scores = {}
        # handle add
        us, vs = np.where(add_scores > 0)
        for i in range(len(us)):
            u, v = int(us[i]), int(vs[i])
            a, b = (u, v) if u < v else (v, u)
            score = float(add_scores[u, v])
            # prefer addition ops encoded as positive score, op=1
            prev = edge_scores.get((a, b), (0.0, None))
            if score > prev[0]:
                edge_scores[(a, b)] = (score, 1)
        # handle remove
        us, vs = np.where(remove_scores > 0)
        for i in range(len(us)):
            u, v = int(us[i]), int(vs[i])
            a, b = (u, v) if u < v else (v, u)
            score = float(remove_scores[u, v])
            prev = edge_scores.get((a, b), (0.0, None))
            if score > prev[0]:
                edge_scores[(a, b)] = (score, 0)

        if len(edge_scores) == 0:
            # nothing beneficial any more
            break

        # sort edges by score desc
        sorted_edges = sorted(edge_scores.items(), key=lambda kv: -kv[1][0])
        scores = np.array([v[0] for _, v in sorted_edges], dtype=np.float32)
        mean_pos = float(scores.mean()) if scores.size > 0 else 0.0
        # thresholding factor to avoid tiny updates
        threshold_factor = 0.5
        acceptance_threshold = threshold_factor * mean_pos

        # pick top-K up to max_changes_per_iter and above threshold
        changes = 0
        for ((u, v), (score, op)) in sorted_edges:
            if changes >= max_changes_per_iter:
                break
            if score < acceptance_threshold:
                continue
            if op == 1:
                raw_adj_mutable[u, v] = 1.0
                raw_adj_mutable[v, u] = 1.0
            else:
                raw_adj_mutable[u, v] = 0.0
                raw_adj_mutable[v, u] = 0.0
            changes += 1

        if changes == 0:
            # no accepted changes this iter
            break

    reshaped = raw_adj_mutable.tocsr()
    reshaped.eliminate_zeros()
    return reshaped, abnormal_train