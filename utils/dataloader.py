import json
import os
import pickle as pkl
from typing import Any, Dict, List

import numpy as np

DEFAULT_DATASET_ROOT = os.environ.get("CONTINUUM_DATA_ROOT", "./datasets")
path_dataset = os.path.abspath(os.path.expanduser(DEFAULT_DATASET_ROOT))


def set_dataset_root(root: str) -> None:
    """Set the dataset root used by all loader helpers."""
    global path_dataset
    path_dataset = os.path.abspath(os.path.expanduser(root))


def dataset_root() -> str:
    return path_dataset


def _dataset_path(*parts: str) -> str:
    return os.path.join(path_dataset, *parts)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pkl.load(f)


def _graph_file_sort_key(filename: str):
    digits = "".join(ch for ch in filename if ch.isdigit())
    return (int(digits) if digits else -1, filename)


def preload_entity_level_dataset(name: str) -> None:
    path = _dataset_path(name)
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.exists(metadata_path):
        return

    malicious = _load_pickle(os.path.join(path, "malicious.pkl"))
    n_train = len(os.listdir(os.path.join(path, "train")))
    n_test = len(os.listdir(os.path.join(path, "test")))
    g = _load_pickle(os.path.join(path, "train", "graph0", "graph0.pkl"))

    metadata = {
        "node_feature_dim": len(g.ndata["attr"][0]),
        "edge_feature_dim": len(g.edata["attr"][0]),
        "malicious": np.asarray(malicious[0] if isinstance(malicious, tuple) else malicious).tolist(),
        "n_train": n_train,
        "n_test": n_test,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata(name: str) -> Dict[str, Any]:
    preload_entity_level_dataset(name)
    with open(_dataset_path(name, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_entity_level_dataset(name: str, split: str, index: int, snapshot: int, device):
    preload_entity_level_dataset(name)
    graphs = []
    graph_dir = _dataset_path(name, split, f"graph{index}")
    for i in range(snapshot):
        graphs.append(_load_pickle(os.path.join(graph_dir, f"graph{i}.pkl")).to(device))
    return graphs


def get_labels(name: str) -> List[int]:
    if name == "wget":
        return [1] * 25 + [0] * 125
    if name == "streamspot":
        return [0] * 300 + [1] * 100 + [0] * 200
    if name == "SC2":
        return [0] * 125 + [1] * 25
    if name == "Unicorn-Cadets":
        return [0] * 109 + [1] * 3
    if name == "wget-long":
        return [0] * 100 + [1] * 5
    if name == "clearscope-e3":
        return [0] * 44 + [1] * 50
    raise ValueError(f"未知图级数据集：{name}")


def load_data(name: str) -> Dict[str, Any]:
    if name == "wget":
        n, n_dim, e_dim = 150, 14, 4
        train_dataset = list(range(50, 150))
        validation_dataset = list(range(50))
        test_dataset = list(range(50))
    elif name == "streamspot":
        n, n_dim, e_dim = 600, 8, 26
        train_dataset = list(range(300))
        validation_dataset = list(range(300, 350)) + list(range(500, 550))
        test_dataset = list(range(300, 400)) + list(range(400, 500)) + list(range(500, 600))
    elif name == "SC2":
        n_dim = len(_load_pickle(_dataset_path("SC2", "node.pkl")).keys())
        e_dim = len(_load_pickle(_dataset_path("SC2", "edge.pkl")).keys())
        n = 150
        train_dataset = list(range(100))
        validation_dataset = list(range(100, 150))
        test_dataset = list(range(100, 150))
    elif name in ["Unicorn-Cadets", "wget-long", "clearscope-e3"]:
        n_dim = len(_load_pickle(_dataset_path(name, "node.pkl")).keys())
        e_dim = len(_load_pickle(_dataset_path(name, "edge.pkl")).keys())
        if name == "Unicorn-Cadets":
            n, train_dataset = 112, list(range(70))
        elif name == "wget-long":
            n, train_dataset = 105, list(range(70))
        else:
            n, train_dataset = 94, list(range(30))
        validation_dataset = list(range(train_dataset[-1], n))
        test_dataset = validation_dataset
    else:
        raise ValueError(f"未知图级数据集：{name}")

    full_dataset_index = list(range(n))
    return {
        "dataset": full_dataset_index,
        "train_index": train_dataset,
        "test_index": test_dataset,
        "validation_index": validation_dataset,
        "full_index": full_dataset_index,
        "n_feat": n_dim,
        "e_feat": e_dim,
        "labels": get_labels(name),
    }


def load_graph(index: int, name: str, device):
    if name == "wget":
        path = _dataset_path("wget", "cache", f"graph{index}")
    elif name == "streamspot":
        path = _dataset_path("streamspot", "cache", f"graph{index}")
    elif name == "SC2":
        if index < 125:
            path = _dataset_path("SC2", "cache", "benign", f"graph{index}")
        else:
            path = _dataset_path("SC2", "cache", "attack", f"graph{index - 125}")
    elif name == "Unicorn-Cadets":
        if index < 109:
            path = _dataset_path("Unicorn-Cadets", "cache", "benign", f"graph{index}")
        else:
            path = _dataset_path("Unicorn-Cadets", "cache", "attack", f"graph{index - 109}")
    elif name == "wget-long":
        if index < 100:
            path = _dataset_path("wget-long", "cache", "benign", f"graph{index}")
        else:
            path = _dataset_path("wget-long", "cache", "attack", f"graph{index - 100}")
    elif name == "clearscope-e3":
        if index < 44:
            path = _dataset_path("clearscope-e3", "cache", "benign", f"graph{index}")
        else:
            path = _dataset_path("clearscope-e3", "cache", "attack", f"graph{index - 44}")
    else:
        raise ValueError(f"未知图级数据集：{name}")

    graphs = []
    for fname in sorted(os.listdir(path), key=_graph_file_sort_key):
        if fname.endswith(".pkl"):
            graphs.append(_load_pickle(os.path.join(path, fname)).to(device))
    return graphs
