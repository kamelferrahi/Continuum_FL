import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from model.model import STGNN_AutoEncoder
from model.train_entity import entity_level_train
from model.train_graph import batch_level_train
from utils.config import GRAPH_LEVEL_DATASETS, build_args
from utils.dataloader import (
    load_data,
    load_entity_level_dataset,
    load_metadata,
    set_dataset_root,
)
from utils.utils import create_optimizer, set_random_seed


def resolve_device(device_arg: str) -> torch.device:
    """Return a torch.device from auto/cpu/cuda/cuda:N style arguments."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。请改用 --device cpu。")
    return torch.device(device_arg)


def infer_dimensions(dataset_name: str) -> Tuple[int, int, bool]:
    """Infer node/edge feature dimensions and whether to concatenate all hidden GAT layers."""
    if dataset_name in GRAPH_LEVEL_DATASETS:
        metadata = load_data(dataset_name)
        return metadata["n_feat"], metadata["e_feat"], True

    metadata = load_metadata(dataset_name)
    return metadata["node_feature_dim"], metadata["edge_feature_dim"], False


def default_epochs(dataset_name: str) -> int:
    return 6 if dataset_name in GRAPH_LEVEL_DATASETS else 50


def default_gnn_layers(dataset_name: str) -> int:
    if dataset_name in GRAPH_LEVEL_DATASETS:
        return 3 if dataset_name == "SC2" else 5
    return 4 if dataset_name == "cadets-e3" else 3


def build_model(args, device: torch.device) -> STGNN_AutoEncoder:
    n_node_feat, n_edge_feat, use_all_hidden = infer_dimensions(args.dataset)
    epochs = args.epochs if args.epochs is not None else default_epochs(args.dataset)
    gnn_layers = args.gnn_layers if args.gnn_layers is not None else default_gnn_layers(args.dataset)

    args.n_dim = n_node_feat
    args.e_dim = n_edge_feat
    args.max_epoch = epochs

    model = STGNN_AutoEncoder(
        n_node_feat,
        n_edge_feat,
        args.out_dim,
        args.out_dim,
        gnn_layers,
        args.heads,
        device,
        args.snapshot,
        args.activation,
        args.feat_drop,
        args.negative_slope,
        True,
        args.norm,
        args.pooling,
        loss_fn=args.loss_fn,
        alpha_l=args.alpha_l,
        use_all_hidden=use_all_hidden,
    )
    return model.to(device)


def checkpoint_path(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    return os.path.join(args.checkpoint_dir, f"checkpoint-{args.dataset}.pt")


def train_graph_level(args, model, device):
    dataset = load_data(args.dataset)
    train_ids = dataset["train_index"]
    train_loader = DataLoader(train_ids, batch_size=args.batch_size, shuffle=True)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    logging.info("开始图级训练：dataset=%s, train_graphs=%s, epochs=%s", args.dataset, len(train_ids), args.max_epoch)
    return batch_level_train(
        model,
        train_loader,
        optimizer,
        args.max_epoch,
        device,
        args.n_dim,
        args.e_dim,
        args.dataset,
        validation=args.validation,
    )


def evaluate_graph_level(args, model, device):
    return batch_level_evaluation(model, model.pooler, device, ["knn"], args.dataset, args.n_dim, args.e_dim)


def train_entity_level(args, model, device):
    metadata = load_metadata(args.dataset)
    train_ids = list(range(metadata["n_train"]))
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    logging.info("开始实体级训练：dataset=%s, train_graphs=%s, epochs=%s", args.dataset, len(train_ids), args.max_epoch)
    return entity_level_train(model, args.snapshot, optimizer, args.max_epoch, device, args.dataset, train_ids)


def _normalize_malicious_index(raw_malicious):
    """Support metadata produced by both pickle tuples and JSON lists."""
    if isinstance(raw_malicious, (list, tuple)) and raw_malicious:
        first = raw_malicious[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return np.asarray(first, dtype=np.int64)
    return np.asarray(raw_malicious, dtype=np.int64)


def evaluate_entity_level(args, model, device):
    metadata = load_metadata(args.dataset)
    malicious = _normalize_malicious_index(metadata["malicious"])
    n_train = metadata["n_train"]
    n_test = metadata["n_test"]

    logging.info("生成训练集节点嵌入：%s 个训练图", n_train)
    with torch.no_grad():
        x_train = []
        for i in range(n_train):
            graphs = load_entity_level_dataset(args.dataset, "train", i, args.snapshot, device)
            x_train.append(model.embed(graphs).cpu().detach().numpy())
            del graphs
        x_train = np.concatenate(x_train, axis=0)

    logging.info("生成测试集节点嵌入：%s 个测试图", n_test)
    x_test = []
    skip_benign = 0
    with torch.no_grad():
        for i in range(n_test):
            graphs = load_entity_level_dataset(args.dataset, "test", i, args.snapshot, device)
            if i != n_test - 1:
                skip_benign += graphs[0].number_of_nodes()
            x_test.append(model.embed(graphs).cpu().detach().numpy())
            del graphs
        x_test = np.concatenate(x_test, axis=0)

    y_test = np.zeros(x_test.shape[0])
    y_test[malicious] = 1.0

    test_idx = [idx for idx in range(x_test.shape[0]) if idx >= skip_benign or y_test[idx] == 1.0]
    result_x_test = x_test[test_idx]
    result_y_test = y_test[test_idx]

    return evaluate_entity_level_using_knn(args.dataset, x_train, result_x_test, result_y_test)[:2]


def run(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.eval_result_dir, exist_ok=True)

    if args.data_root:
        set_dataset_root(args.data_root)

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("使用设备：%s", device)

    model = build_model(args, device)
    ckpt = checkpoint_path(args)

    if args.eval_only:
        logging.info("跳过训练，加载检查点：%s", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        if args.dataset in GRAPH_LEVEL_DATASETS:
            model = train_graph_level(args, model, device)
        else:
            model = train_entity_level(args, model, device)
        torch.save(model.state_dict(), ckpt)
        logging.info("已保存检查点：%s", ckpt)

    model.eval()
    if args.dataset in GRAPH_LEVEL_DATASETS:
        auc_score, auc_std = evaluate_graph_level(args, model, device)
    else:
        auc_score, auc_std = evaluate_entity_level(args, model, device)

    result_path = os.path.join(args.result_dir, f"local-{args.dataset}.pt")
    torch.save(model.state_dict(), result_path)
    logging.info("评估完成：AUC=%s, STD=%s", auc_score, auc_std)
    logging.info("最终模型已保存：%s", result_path)


if __name__ == "__main__":
    parsed_args = build_args()
    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parsed_args)
