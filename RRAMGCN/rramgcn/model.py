import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from copy import deepcopy
from rramgcn.reshape import graphreshape
from rramgcn.deepgcn import adjPrecess, DeepGCN
from rramgcn.utils import csr_to_torch_sparse_tensor, set_device, normalize_adj_sym, eliminate_zeros_sparse, torch_sparse_to_dense
device = set_device()
class RRAMGCNTrainer:
    def __init__(self,
                 device=None,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.6,
                 epochs=200,
                 sim_threshold=0.6,
                 structure_diff_weight=0.7,
                 lr=0.005,
                 weight_decay=0.002,
                 patience=40,
                 inner_finetune_steps=3,
                 candidate_k=15,
                 max_candidates=6000,
                 attention_tau=0.6,
                 clamp_sim_min=True,
                 attn_drop=0.25,
                 importance_alpha=0.7,
                 importance_beta=0.3,
                 max_changes_per_iter=2,
                 recompute_every=10):
        self.device = set_device(device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.sim_threshold = sim_threshold
        self.structure_diff_weight = structure_diff_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inner_finetune_steps = inner_finetune_steps
        self.candidate_k = candidate_k
        self.max_candidates = max_candidates
        self.attention_tau = attention_tau
        self.clamp_sim_min = clamp_sim_min
        self.attn_drop = attn_drop
        self.importance_alpha = importance_alpha
        self.importance_beta = importance_beta
        self.max_changes_per_iter = max_changes_per_iter
        self.recompute_every = recompute_every

    def _compute_degree(self, adj_sp):
        if sp.issparse(adj_sp):
            deg = np.array(adj_sp.sum(axis=1)).flatten()
        else:
            deg = np.sum(adj_sp, axis=1)
        return deg

    def _prepare_multi_channel_adjs(self, raw_adj_sp, features_np):
        raw_adj = raw_adj_sp.tocsr().copy()
        feat = features_np.astype(np.float32)
        n = raw_adj.shape[0]
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(feat, feat).astype(np.float32)
        orig_deg = self._compute_degree(raw_adj)
        adj_coo = raw_adj.tocoo()
        rows, cols, data = adj_coo.row.copy(), adj_coo.col.copy(), adj_coo.data.copy()
        sim_mask = sim_matrix[rows, cols] < self.sim_threshold
        edge_diff = np.zeros(len(rows), dtype=np.float32)
        for idx in range(len(rows)):
            if sim_mask[idx]:
                edge_diff[idx] = abs(orig_deg[rows[idx]] - orig_deg[cols[idx]])
        max_ed = edge_diff.max() if edge_diff.size > 0 else 0.0
        if max_ed > 0:
            edge_diff = edge_diff / (max_ed + 1e-12)
        remove_mask = sim_mask & (edge_diff > self.structure_diff_weight)
        data[remove_mask] = 0.0
        A_L = sp.csr_matrix((data, (rows, cols)), shape=raw_adj.shape)
        A_L = (A_L + A_L.T).tocsr()
        A_L.data[A_L.data > 1.0] = 1.0
        A_L.eliminate_zeros()
        A_H = raw_adj - A_L
        A_H = (A_H + A_H.T).tocsr()
        A_H.data[A_H.data > 1.0] = 1.0
        A_H.eliminate_zeros()
        mid = A_L.dot(A_L).astype(np.float32)
        mid.eliminate_zeros()
        A_L_norm = normalize_adj_sym(A_L, add_self_loops=True)
        A_H_norm = normalize_adj_sym(A_H, add_self_loops=True)
        mid_norm = normalize_adj_sym(mid, add_self_loops=True)
        A_L_t = csr_to_torch_sparse_tensor(A_L_norm, device=self.device)
        A_H_t = csr_to_torch_sparse_tensor(A_H_norm, device=self.device)
        mid_t = csr_to_torch_sparse_tensor(mid_norm, device=self.device)
        return A_L_t, A_H_t, mid_t, A_L, A_H, mid, sim_matrix

    def _get_candidate_mask(self, important_nodes, sim_matrix, A_L_sp, candidate_k=None):
        n = A_L_sp.shape[0]
        cand_set = set()
        if candidate_k is None:
            deg = np.array(A_L_sp.sum(axis=1)).flatten()
            avg_deg = float(np.mean(deg)) if deg.size > 0 else 1.0
            candidate_k = min(15, max(5, int(avg_deg / 2)))
        for u in important_nodes:
            sims = sim_matrix[u]
            topk = np.argpartition(-sims, min(candidate_k, len(sims)-1))[:candidate_k]
            for v in topk:
                if v == u:
                    continue
                a, b = (int(u), int(v))
                if a > b:
                    a, b = b, a
                cand_set.add((a, b))
            nbrs = A_L_sp[u].nonzero()[1].tolist()
            for v in nbrs:
                if v == u:
                    continue
                a, b = (int(u), int(v))
                if a > b:
                    a, b = b, a
                cand_set.add((a, b))
        cand_list = list(cand_set)
        if len(cand_list) > self.max_candidates:
            cand_list = cand_list[:self.max_candidates]
        return cand_list

    def fit_and_eval(self, features, labels, perturbed_adj, split_train, split_val, split_unlabeled, dataset_name='dataset', target_nodes=None):
        if sp.issparse(perturbed_adj):
            raw_adj = perturbed_adj.tocsr().astype(np.float32).copy()
        else:
            raw_adj = sp.csr_matrix(perturbed_adj).astype(np.float32)
        print("[RRAMGCN] Stage 1: identify important nodes with Reshape...")
        reshaped_adj, important_nodes = graphreshape(raw_adj, features, labels,
                                                     split_train, split_val, split_unlabeled,
                                                     n_sample=1, lr=5e-4, h_dim=32, threshold=0.7,
                                                     importance_alpha=self.importance_alpha,
                                                     importance_beta=self.importance_beta,
                                                     candidate_k=self.candidate_k,
                                                     max_changes_per_iter=self.max_changes_per_iter,
                                                     recompute_every=self.recompute_every,
                                                     max_candidates=self.max_candidates)
        print(f"[RRAMGCN] Found {len(important_nodes)} important nodes.")
        A_L_t, A_H_t, mid_t, A_L_sp, A_H_sp, mid_sp, sim_matrix = self._prepare_multi_channel_adjs(reshaped_adj, features.cpu().numpy() if isinstance(features, torch.Tensor) else features)
        data = {'low_adj': A_L_t, 'high_adj': A_H_t, 'mid_adj': mid_t}
        nfeat = features.shape[1]
        nclass = int(labels.max().item()) + 1 if torch.is_tensor(labels) else int(labels.max()) + 1
        model = DeepGCN(nfeat, self.hidden_dim, nclass, num_layers=self.num_layers, dropout=self.dropout,
                        attention_tau=self.attention_tau, clamp_sim_min=self.clamp_sim_min, attn_drop=self.attn_drop)
        model.to(self.device)
        features_t = features.to(self.device)
        labels_t = labels.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = 0.0
        best_state = None
        patience = self.patience
        cand_pairs = self._get_candidate_mask(important_nodes, sim_matrix, A_L_sp, candidate_k=self.candidate_k)
        print(f"[RRAMGCN] Candidate pairs for structural updates: {len(cand_pairs)}")
        for ep in range(self.epochs):
            model.train()
            if self.recompute_every is not None and self.recompute_every > 0 and (ep % self.recompute_every == 0) and ep > 0:
                cand_pairs = self._get_candidate_mask(important_nodes, sim_matrix, A_L_sp, candidate_k=self.candidate_k)
            for layer_idx in range(self.num_layers):
                inner_steps = max(1, self.inner_finetune_steps)
                inner_opt = torch.optim.Adam(model.parameters(), lr=max(1e-3, self.lr*0.5))
                for _ in range(inner_steps):
                    model.train()
                    out, _ = model(features_t, data)
                    loss_in = F.nll_loss(out[split_train], labels_t[split_train])
                    inner_opt.zero_grad()
                    loss_in.backward()
                    inner_opt.step()
                A_L_norm = normalize_adj_sym(A_L_sp, add_self_loops=True)
                A_H_norm = normalize_adj_sym(A_H_sp, add_self_loops=True)
                A_L_norm_t = csr_to_torch_sparse_tensor(A_L_norm, device=self.device)
                A_H_norm_t = csr_to_torch_sparse_tensor(A_H_norm, device=self.device)
                A_L_dense = A_L_norm_t.to_dense().to(self.device)
                A_L_dense = A_L_dense.clone().detach().requires_grad_(True)
                temp_data = {'low_adj': A_L_dense, 'high_adj': A_H_norm_t, 'mid_adj': mid_t}
                model.eval()
                out_meta, _ = model(features_t, temp_data)
                if split_val is None or len(split_val) == 0:
                    meta_idx = split_train
                else:
                    meta_idx = split_val
                loss_meta = F.nll_loss(out_meta[meta_idx], labels_t[meta_idx])
                if A_L_dense.grad is not None:
                    A_L_dense.grad.zero_()
                loss_meta.backward()
                if A_L_dense.grad is None:
                    try:
                        g = torch.autograd.grad(loss_meta, A_L_dense, retain_graph=False, create_graph=False)
                        grad_adj = g[0].detach().cpu().numpy() if (g is not None and g[0] is not None) else None
                    except Exception:
                        grad_adj = None
                else:
                    grad_adj = A_L_dense.grad.detach().cpu().numpy()
                if grad_adj is None:
                    print("[RRAMGCN] Warning: grad_adj is None at epoch", ep, "layer", layer_idx, "— skipping structural update for this layer.")
                    continue
                base = A_L_sp.toarray()
                N = base.shape[0]
                mask = np.zeros_like(grad_adj, dtype=bool)
                imp_nodes = important_nodes
                mask[np.ix_(imp_nodes, np.arange(N))] = True
                mask[np.ix_(np.arange(N), imp_nodes)] = True
                cand_mask = np.zeros_like(grad_adj, dtype=bool)
                for (u, v) in cand_pairs:
                    cand_mask[u, v] = True
                    cand_mask[v, u] = True
                final_mask = mask & cand_mask
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

                importance = self.importance_alpha * deg_sum_norm + self.importance_beta * (1.0 - common_ratio)
                importance = np.maximum(importance, 0.0)

                importance_masked = np.where(final_mask, importance, 0.0)
                grad_masked = np.where(final_mask, grad_adj, 0.0)

                add_candidates = (base <= 0.5) & (grad_masked < 0) & final_mask
                add_scores = np.where(add_candidates, (-grad_masked) * importance_masked, 0.0)

                remove_candidates = (base > 0.5) & (grad_masked > 0) & final_mask
                remove_scores = np.where(remove_candidates, grad_masked * importance_masked, 0.0)

                edge_scores = {}
                us, vs = np.where(add_scores > 0)
                for i in range(len(us)):
                    u, v = int(us[i]), int(vs[i])
                    a, b = (u, v) if u < v else (v, u)
                    score = float(add_scores[u, v])
                    prev = edge_scores.get((a, b), (0.0, None))
                    if score > prev[0]:
                        edge_scores[(a, b)] = (score, 1)
                us, vs = np.where(remove_scores > 0)
                for i in range(len(us)):
                    u, v = int(us[i]), int(vs[i])
                    a, b = (u, v) if u < v else (v, u)
                    score = float(remove_scores[u, v])
                    prev = edge_scores.get((a, b), (0.0, None))
                    if score > prev[0]:
                        edge_scores[(a, b)] = (score, 0)

                if len(edge_scores) == 0:
                    continue

                sorted_edges = sorted(edge_scores.items(), key=lambda kv: -kv[1][0])
                scores = np.array([v[0] for _, v in sorted_edges], dtype=np.float32)
                mean_pos = float(scores.mean()) if scores.size > 0 else 0.0
                threshold_factor = 0.5
                acceptance_threshold = threshold_factor * mean_pos

                changes = 0
                for ((u, v), (score, op)) in sorted_edges:
                    if changes >= self.max_changes_per_iter:
                        break
                    if score < acceptance_threshold:
                        continue
                    if op == 1:
                        A_L_sp[u, v] = 1.0
                        A_L_sp[v, u] = 1.0
                        A_H_sp[u, v] = 0.0
                        A_H_sp[v, u] = 0.0
                    else:
                        A_L_sp[u, v] = 0.0
                        A_L_sp[v, u] = 0.0
                        A_H_sp[u, v] = 1.0
                        A_H_sp[v, u] = 1.0
                    A_L_sp.eliminate_zeros()
                    A_H_sp.eliminate_zeros()
                    changes += 1

                if changes > 0:
                    A_L_t = csr_to_torch_sparse_tensor(normalize_adj_sym(A_L_sp, add_self_loops=True), device=self.device)
                    A_H_t = csr_to_torch_sparse_tensor(normalize_adj_sym(A_H_sp, add_self_loops=True), device=self.device)
                    mid_sp = A_L_sp.dot(A_L_sp).astype(np.float32)
                    mid_sp.eliminate_zeros()
                    mid_t = csr_to_torch_sparse_tensor(normalize_adj_sym(mid_sp, add_self_loops=True), device=self.device)
                    data['low_adj'] = A_L_t
                    data['high_adj'] = A_H_t
                    data['mid_adj'] = mid_t

            model.train()
            optimizer.zero_grad()
            out_train, _ = model(features_t, data)
            loss = F.nll_loss(out_train[split_train], labels_t[split_train])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_eval, _ = model(features_t, data)
                pred_val = out_eval[split_val].argmax(dim=1)
                val_acc = (pred_val == labels_t[split_val]).float().mean().item()
                pred_test = out_eval[split_unlabeled].argmax(dim=1)
                test_acc = (pred_test == labels_t[split_unlabeled]).float().mean().item()

            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_state = deepcopy(model.state_dict())
                patience = self.patience
            else:
                patience -= 1

            if ep % 10 == 0:
                print(f"[Epoch {ep}] train_loss={loss.item():.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

            if patience <= 0:
                print(f"[RRAMGCN] Early stopping at epoch {ep}")
                break

        print("[RRAMGCN] Final evaluation with best model...")
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_final, _ = model(features_t, data)
            if target_nodes is not None and len(target_nodes) > 0:
                tnodes = torch.tensor(target_nodes, device=self.device)
                pred_final = out_final[tnodes].argmax(dim=1)
                final_acc = (pred_final == labels_t[tnodes]).float().mean().item()
            else:
                pred_final = out_final[split_unlabeled].argmax(dim=1)
                final_acc = (pred_final == labels_t[split_unlabeled]).float().mean().item()
        print(f"[RRAMGCN] Final acc: {final_acc:.4f}")
        return final_acc, data, important_nodes