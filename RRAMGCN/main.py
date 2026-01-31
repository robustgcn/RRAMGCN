import os
import torch
import numpy as np
import time
import json
import scipy.sparse as sp
from rramgcn.utils import set_seed
from rramgcn.model import RRAMGCNTrainer

def _to_scipy_adj(adj):
    if sp.issparse(adj):
        return adj.tocsr()
    if isinstance(adj, np.ndarray):
        return sp.csr_matrix(adj)
    if isinstance(adj, torch.Tensor):
        return sp.csr_matrix(adj.cpu().numpy())
    raise ValueError(f"Unsupported adj format: {type(adj)}")

def save_results(results, out_dir):
    for dataset, data_res in results.items():
        result_file = os.path.join(out_dir, f"{dataset}_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                try:
                    existing_res = json.load(f)
                except Exception:
                    existing_res = {}
            existing_res.update(data_res)
            data_res = existing_res
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data_res, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至 {out_dir}")

def _load_original_data(dataset, attacked_root='attacked_graphs'):
    """加载原始数据（从 attacked_graphs/<dataset>/original_data.npz ）"""
    data_path = os.path.join(attacked_root, dataset, 'original_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"原始数据文件缺失: {data_path}")

    with np.load(data_path, allow_pickle=True) as f:
        A_orig = f['adj'].item() if isinstance(f['adj'], np.ndarray) else f['adj']
        features = f['features'].item() if isinstance(f['features'], np.ndarray) else f['features']
        labels_np = f['labels']
        split_train = f['idx_train']
        split_val = f['idx_val']
        split_unlabeled = f.get('idx_test', f.get('idx_unlabeled', f['idx_val']))
    A_orig = _to_scipy_adj(A_orig)
    return labels_np, features, split_train, split_val, split_unlabeled, A_orig

def _load_perturbed_adj(dataset, perturb_ratio, attacked_root='attacked_graphs'):
    """加载攻击后的邻接矩阵"""
    ratio_str = f"{perturb_ratio:.2f}"
    data_path = os.path.join(attacked_root, dataset, f'attacked_p{ratio_str}.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"攻击文件缺失: {data_path}")
    with np.load(data_path, allow_pickle=True) as f:
        adj_attack = f['adj_attack'].item() if isinstance(f['adj_attack'], np.ndarray) else f['adj_attack']
    return _to_scipy_adj(adj_attack)

DATASETS = ['cora']
perturb_ratios = [0.05]
RRAMGCN_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.6,
    'structure_diff_weight': 0.7,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000
}

ATTACKED_DIR = 'attacked_graphs'
OUT_DIR = 'results_rramgcn'
os.makedirs(OUT_DIR, exist_ok=True)
use_cuda = torch.cuda.is_available()
model_device = torch.device("cuda" if use_cuda else "cpu")
attack_device = torch.device("cpu")
print(f"设备配置: 模型训练={model_device}, 攻击执行={attack_device}")
print(f"RAM-GCN超参数: {RRAMGCN_PARAMS}")
print(f"攻击文件根目录: {ATTACKED_DIR}")

def run_all():
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        print(f"\n=== 开始处理数据集: {dataset} ===")
        # 1. 加载原始数据
        try:
            labels_np, features, split_train, split_val, split_unlabeled, A_orig = _load_original_data(
                dataset=dataset, attacked_root=ATTACKED_DIR
            )
            if sp.issparse(features):
                feats_np = features.toarray()
                features = torch.from_numpy(feats_np).float().to(model_device)
            elif isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float().to(model_device)
            elif isinstance(features, torch.Tensor):
                features = features.to(model_device)
            else:
                raise TypeError(f"不支持的特征格式: {type(features)}")

            labels = torch.tensor(labels_np, dtype=torch.long, device=model_device)
            split_train = np.array(split_train, dtype=int)
            split_val = np.array(split_val, dtype=int)
            split_unlabeled = np.array(split_unlabeled, dtype=int)
            print(f"原始数据加载成功: 节点数={A_orig.shape[0]}, 边数={int(A_orig.sum()//2)}, 特征维度={features.shape[1]}")
            print(f"split sizes -> train:{len(split_train)} val:{len(split_val)} test:{len(split_unlabeled)}")
            print("overlaps -> train∩val:", len(set(split_train) & set(split_val)),
                  "train∩test:", len(set(split_train) & set(split_unlabeled)),
                  "val∩test:", len(set(split_val) & set(split_unlabeled)))
        except Exception as e:
            print(f" 数据集 {dataset} 加载失败: {str(e)}")
            results[dataset]['error'] = f"数据加载失败: {str(e)}"
            save_results(results, OUT_DIR)
            continue
        # 2. 扰动实验
        attack_type = 'metattack'
        for ratio in perturb_ratios:
            ratio_key = str(ratio)
            results[dataset][ratio_key] = {}
            print(f"\n-> 当前扰动比例: {ratio}（攻击类型: {attack_type}）")
            try:
                start_time = time.time()
                if ratio == 0.0:
                    perturbed_adj = A_orig
                    print(f"  使用原始邻接矩阵（无扰动）")
                else:
                    perturbed_adj = _load_perturbed_adj(dataset, ratio, ATTACKED_DIR)
                    print(f"  加载扰动邻接矩阵完成")
                trainer = RRAMGCNTrainer(**RRAMGCN_PARAMS, device=str(model_device))
                def_acc = trainer.fit_and_eval(
                    features=features,
                    labels=labels,
                    perturbed_adj=perturbed_adj,
                    split_train=split_train,
                    split_val=split_val,
                    split_unlabeled=split_unlabeled,
                    dataset_name=dataset
                )
                elapsed_time = time.time() - start_time
                if isinstance(def_acc, (list, tuple)):
                    def_acc_val = def_acc[0]
                else:
                    def_acc_val = def_acc
                def_acc_val = float(def_acc_val)
                def_acc_val = round(def_acc_val, 4)
                results[dataset][ratio_key][attack_type] = def_acc_val
                print(f" {attack_type} @ 扰动{ratio}: 防御准确率={def_acc_val:.4f}, 耗时={elapsed_time:.2f}秒")
            except Exception as e:
                error_msg = str(e)[:200]
                print(f" 执行失败: {error_msg}")
                results[dataset][ratio_key][attack_type] = f"失败: {error_msg}"
                continue
        save_results(results, OUT_DIR)
        print(f"=== 数据集 {dataset} 处理完成 ===")
    print("\n 所有数据集实验处理完成！")
    return results
if __name__ == "__main__":
    set_seed(42, use_cuda)
    run_all()