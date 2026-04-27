import argparse
import json
import os
import pickle as pkl
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dgl
import networkx as nx
import torch
from tqdm import tqdm


DATASET_FILES = {
    "trace-e3": {
        "train": [
            "ta1-trace-e3-official-1.json",
            "ta1-trace-e3-official-1.json.1",
            "ta1-trace-e3-official-1.json.2",
            "ta1-trace-e3-official-1.json.3",
        ],
        "test": [
            "ta1-trace-e3-official-1.json",
            "ta1-trace-e3-official-1.json.1",
            "ta1-trace-e3-official-1.json.2",
            "ta1-trace-e3-official-1.json.3",
            "ta1-trace-e3-official-1.json.4",
        ],
    },
    "theia-e3": {
        "train": [
            "ta1-theia-e3-official-6r.json",
            "ta1-theia-e3-official-6r.json.1",
            "ta1-theia-e3-official-6r.json.2",
            "ta1-theia-e3-official-6r.json.3",
        ],
        "test": ["ta1-theia-e3-official-6r.json.8"],
    },
    "cadets-e3": {
        "train": [
            "ta1-cadets-e3-official.json",
            "ta1-cadets-e3-official.json.1",
            "ta1-cadets-e3-official.json.2",
            "ta1-cadets-e3-official-2.json.1",
        ],
        "test": ["ta1-cadets-e3-official-2.json"],
    },
}

UUID_RE = re.compile(r'uuid\\":\\"(.*?)\\"')
SRC_RE = re.compile(r'subject\\":\\{\\"com.bbn.tc.schema.avro.cdm18.UUID\\":\\"(.*?)\\"\\}')
DST1_RE = re.compile(r'predicateObject\\":\\{\\"com.bbn.tc.schema.avro.cdm18.UUID\\":\\"(.*?)\\"\\}')
DST2_RE = re.compile(r'predicateObject2\\":\\{\\"com.bbn.tc.schema.avro.cdm18.UUID\\":\\"(.*?)\\"\\}')
TYPE_RE = re.compile(r'type\\":\\"(.*?)\\"')
TIME_RE = re.compile(r'timestampNanos\\":(.*?),')
FILE_NAME_RE = re.compile(r'map\\":\\{\\"path\\":\\"(.*?)\\"')
PROCESS_NAME_RE = re.compile(r'map\\":\\{\\"name\\":\\"(.*?)\\"')
NETFLOW_NAME_RE = re.compile(r'remoteAddress\\":\\"(.*?)\\"')

SKIP_RECORD_MARKERS = (
    "com.bbn.tc.schema.avro.cdm18.Event",
    "com.bbn.tc.schema.avro.cdm18.Host",
    "com.bbn.tc.schema.avro.cdm18.TimeMarker",
    "com.bbn.tc.schema.avro.cdm18.StartMarker",
    "com.bbn.tc.schema.avro.cdm18.UnitDependency",
    "com.bbn.tc.schema.avro.cdm18.EndMarker",
)


class DarpaE3Preprocessor:
    """Preprocess DARPA Transparent Computing E3 node-level datasets.

    This module is adapted from the original CONTINUUM repository and cleaned up so it can be
    used in the local non-FL version of the project. It implements the paper's data construction
    pipeline: raw CDM JSON -> edge tables -> NetworkX provenance graph -> timestamp snapshots
    -> compressed edge feature vectors -> DGLGraph pickle files.
    """

    def __init__(self, dataset: str, data_root: str, snapshot: int = 1, force: bool = False, seed: int = 0):
        if dataset not in DATASET_FILES:
            raise ValueError(f"Unsupported DARPA E3 dataset: {dataset}. Choose from {sorted(DATASET_FILES)}")
        if snapshot < 1:
            raise ValueError("snapshot must be >= 1")
        self.dataset = dataset
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.dataset_dir = os.path.join(self.data_root, dataset)
        self.snapshot = snapshot
        self.force = force
        self.rng = random.Random(seed)

        self.node_type_dict: Dict[str, int] = {}
        self.edge_type_dict: Dict[str, int] = {}
        self.id_nodetype_map: Dict[str, str] = {}
        self.id_nodename_map: Dict[str, str] = {}

    @property
    def node_type_count(self) -> int:
        return len(self.node_type_dict)

    @property
    def edge_type_count(self) -> int:
        return len(self.edge_type_dict)

    def _path(self, *parts: str) -> str:
        return os.path.join(self.dataset_dir, *parts)

    def ensure_directories(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self._path("train"), exist_ok=True)
        os.makedirs(self._path("test"), exist_ok=True)

    def required_raw_files(self) -> List[str]:
        files = []
        for split in ("train", "test"):
            for file_name in DATASET_FILES[self.dataset][split]:
                if file_name not in files:
                    files.append(file_name)
        return files

    def validate_inputs(self) -> None:
        missing = []
        for file_name in self.required_raw_files():
            if not os.path.exists(self._path(file_name)):
                missing.append(file_name)
        malicious_path = self._path(f"{self.dataset}.txt")
        if not os.path.exists(malicious_path):
            missing.append(f"{self.dataset}.txt")
        if missing:
            formatted = "\n".join(f"  - {name}" for name in missing)
            raise FileNotFoundError(
                f"Missing required files under {self.dataset_dir}:\n{formatted}\n"
                f"See docs/darpa_e3_preprocessing.md for the expected layout."
            )

    def _record_node(self, uuid: str, subject_type: str, line: str) -> None:
        if uuid == "00000000-0000-0000-0000-000000000000" or subject_type == "SUBJECT_UNIT":
            return
        self.id_nodetype_map[uuid] = subject_type
        if "FILE" in subject_type and FILE_NAME_RE.findall(line):
            self.id_nodename_map[uuid] = FILE_NAME_RE.findall(line)[0]
        elif subject_type == "SUBJECT_PROCESS" and PROCESS_NAME_RE.findall(line):
            self.id_nodename_map[uuid] = PROCESS_NAME_RE.findall(line)[0]
        elif subject_type == "NetFlowObject" and NETFLOW_NAME_RE.findall(line):
            self.id_nodename_map[uuid] = NETFLOW_NAME_RE.findall(line)[0]

    def _infer_subject_type(self, line: str) -> Optional[str]:
        subject_type = TYPE_RE.findall(line)
        if subject_type:
            return subject_type[0]
        if "com.bbn.tc.schema.avro.cdm18.MemoryObject" in line:
            return "MemoryObject"
        if "com.bbn.tc.schema.avro.cdm18.NetFlowObject" in line:
            return "NetFlowObject"
        if "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject" in line:
            return "UnnamedPipeObject"
        return None

    def collect_entity_metadata(self) -> None:
        """Scan raw JSON files once to map entity UUIDs to types and display names."""
        for file_name in self.required_raw_files():
            raw_path = self._path(file_name)
            print(f"[1/5] scanning entities: {raw_path}")
            with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in tqdm(f, unit="lines"):
                    if any(marker in line for marker in SKIP_RECORD_MARKERS):
                        continue
                    uuid_match = UUID_RE.findall(line)
                    if not uuid_match:
                        continue
                    subject_type = self._infer_subject_type(line)
                    if not subject_type:
                        continue
                    self._record_node(uuid_match[0], subject_type, line)

        with open(self._path("types.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_nodetype_map, f, ensure_ascii=False, indent=2)
        with open(self._path("names.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_nodename_map, f, ensure_ascii=False, indent=2)

    def _write_event_edge(self, fw, src_id: str, dst_id: str, edge_type: str, timestamp: str) -> None:
        if src_id not in self.id_nodetype_map or dst_id not in self.id_nodetype_map:
            return
        src_type = self.id_nodetype_map[src_id]
        dst_type = self.id_nodetype_map[dst_id]
        fw.write(f"{src_id}\t{src_type}\t{dst_id}\t{dst_type}\t{edge_type}\t{timestamp}\n")

    def generate_edge_tables(self) -> None:
        """Convert raw CDM JSON events into tab-separated source/destination edge tables."""
        for split in ("train", "test"):
            for file_name in DATASET_FILES[self.dataset][split]:
                txt_path = self._path(f"{file_name}.txt")
                if os.path.exists(txt_path) and not self.force:
                    print(f"[2/5] using cached edge table: {txt_path}")
                    continue
                raw_path = self._path(file_name)
                print(f"[2/5] processing events: {raw_path}")
                with open(raw_path, "r", encoding="utf-8", errors="ignore") as f, open(txt_path, "w", encoding="utf-8") as fw:
                    for line in tqdm(f, unit="lines"):
                        if "com.bbn.tc.schema.avro.cdm18.Event" not in line:
                            continue
                        edge_type = TYPE_RE.findall(line)
                        timestamp = TIME_RE.findall(line)
                        src_id = SRC_RE.findall(line)
                        if not edge_type or not timestamp or not src_id:
                            continue
                        edge_type = edge_type[0]
                        timestamp = timestamp[0]
                        src_id = src_id[0]
                        dst1 = DST1_RE.findall(line)
                        if dst1 and dst1[0] != "null":
                            self._write_event_edge(fw, src_id, dst1[0], edge_type, timestamp)
                        dst2 = DST2_RE.findall(line)
                        if dst2 and dst2[0] != "null":
                            self._write_event_edge(fw, src_id, dst2[0], edge_type, timestamp)

    def load_malicious_entities(self) -> Set[str]:
        malicious_path = self._path(f"{self.dataset}.txt")
        with open(malicious_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _node_type_id(self, node_type: str) -> int:
        if node_type not in self.node_type_dict:
            self.node_type_dict[node_type] = len(self.node_type_dict)
        return self.node_type_dict[node_type]

    def _edge_type_id(self, edge_type: str) -> int:
        if edge_type not in self.edge_type_dict:
            self.edge_type_dict[edge_type] = len(self.edge_type_dict)
        return self.edge_type_dict[edge_type]

    def read_single_graph(self, file_name: str, malicious: Set[str], is_test: bool) -> Tuple[Dict[str, int], nx.MultiDiGraph]:
        """Build one full NetworkX provenance graph from an intermediate edge table."""
        edge_table = self._path(f"{file_name}.txt")
        graph = nx.MultiDiGraph()
        print(f"[3/5] building graph from {edge_table}")

        rows = []
        with open(edge_table, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 6:
                    continue
                src, src_type, dst, dst_type, edge_type, ts = parts
                ts = int(ts)
                if not is_test:
                    # Keep training mostly benign by dropping non-memory malicious entities,
                    # following the original CONTINUUM preprocessing script.
                    if src in malicious and src_type != "MemoryObject":
                        continue
                    if dst in malicious and dst_type != "MemoryObject":
                        continue
                if "READ" in edge_type or "RECV" in edge_type or "LOAD" in edge_type:
                    rows.append((dst, src, dst_type, src_type, edge_type, ts))
                else:
                    rows.append((src, dst, src_type, dst_type, edge_type, ts))
        rows.sort(key=lambda item: item[5])

        node_map: Dict[str, int] = {}
        node_names: List[str] = []
        for src, dst, src_type, dst_type, edge_type, ts in tqdm(rows, unit="edges"):
            src_type_id = self._node_type_id(src_type)
            dst_type_id = self._node_type_id(dst_type)
            edge_type_id = self._edge_type_id(edge_type)
            if src not in node_map:
                node_map[src] = len(node_map)
                node_names.append(src)
                graph.add_node(node_map[src], type=src_type_id)
            if dst not in node_map:
                node_map[dst] = len(node_map)
                node_names.append(dst)
                graph.add_node(node_map[dst], type=dst_type_id)
            graph.add_edge(
                node_map[src],
                node_map[dst],
                key=(node_map[src], node_map[dst], edge_type_id, ts),
                type=edge_type_id,
                time=ts,
            )
        return node_map, graph

    def write_snapshots(self, graphs: Sequence[nx.MultiDiGraph], split: str) -> None:
        """Split graphs by timestamp into snapshots and compress repeated edges."""
        split_dir = self._path(split)
        os.makedirs(split_dir, exist_ok=True)
        for graph_index, graph in enumerate(graphs):
            timestamps = nx.get_edge_attributes(graph, "time")
            edge_types = nx.get_edge_attributes(graph, "type")
            node_types = nx.get_node_attributes(graph, "type")
            if not timestamps:
                raise ValueError(f"Graph {graph_index} in split {split} has no timestamped edges")
            min_time = min(timestamps.values())
            max_time = max(timestamps.values())
            span = max(max_time - min_time, 1)
            interval = span / self.snapshot

            graph_dir = os.path.join(split_dir, f"graph{graph_index}")
            os.makedirs(graph_dir, exist_ok=True)
            print(f"[4/5] writing {self.snapshot} snapshots for {split}/graph{graph_index}")
            for snapshot_index in range(self.snapshot):
                start = min_time + interval * snapshot_index
                end = min_time + interval * (snapshot_index + 1)
                snapshot_graph = nx.DiGraph()
                for node in graph.nodes:
                    snapshot_graph.add_node(node, type=node_types[node])

                for edge in graph.edges(keys=True):
                    ts = timestamps[edge]
                    include = start <= ts < end if snapshot_index < self.snapshot - 1 else start <= ts <= end
                    if not include:
                        continue
                    src, dst, _ = edge
                    edge_type = edge_types[edge]
                    if not snapshot_graph.has_edge(src, dst):
                        attr = [0] * self.edge_type_count
                        attr[edge_type] = 1
                        snapshot_graph.add_edge(src, dst, attr=attr)
                    else:
                        snapshot_graph[src][dst]["attr"][edge_type] += 1

                out_path = os.path.join(graph_dir, f"graph{snapshot_index}.pkl")
                with open(out_path, "wb") as f:
                    pkl.dump(snapshot_graph, f)
                print(
                    f"    snapshot {snapshot_index}: raw_edges={graph.number_of_edges()} "
                    f"compressed_edges={snapshot_graph.number_of_edges()} -> {out_path}"
                )

    def convert_snapshots_to_dgl(self, split: str) -> None:
        """Convert NetworkX snapshot pickles into DGLGraph pickles consumed by main.py."""
        split_dir = self._path(split)
        for graph_name in sorted(os.listdir(split_dir), key=_graph_dir_sort_key):
            graph_dir = os.path.join(split_dir, graph_name)
            if not os.path.isdir(graph_dir):
                continue
            for snapshot_index in range(self.snapshot):
                path = os.path.join(graph_dir, f"graph{snapshot_index}.pkl")
                with open(path, "rb") as f:
                    nx_graph = pkl.load(f)
                dgl_graph = dgl.from_networkx(nx_graph, node_attrs=["type"], edge_attrs=["attr"])
                node_type_tensor = torch.as_tensor(dgl_graph.ndata["type"], dtype=torch.long)
                dgl_graph.ndata["attr"] = torch.nn.functional.one_hot(
                    node_type_tensor, num_classes=self.node_type_count
                ).float()
                dgl_graph.edata["attr"] = torch.as_tensor(dgl_graph.edata["attr"], dtype=torch.float32)
                with open(path, "wb") as f:
                    pkl.dump(dgl_graph, f)

    def write_malicious_files(self, malicious: Set[str], test_node_maps: Sequence[Dict[str, int]]) -> None:
        merged_test_map: Dict[str, int] = {}
        offset = 0
        for node_map in test_node_maps:
            for uuid, local_index in node_map.items():
                if uuid not in merged_test_map:
                    merged_test_map[uuid] = local_index + offset
            offset += len(node_map)

        malicious_indices = []
        malicious_names = []
        names_path = self._path("malicious_names.txt")
        with open(names_path, "w", encoding="utf-8") as f:
            for uuid in sorted(malicious):
                node_type = self.id_nodetype_map.get(uuid)
                if uuid not in merged_test_map:
                    continue
                if node_type in {"MemoryObject", "UnnamedPipeObject"}:
                    continue
                malicious_indices.append(merged_test_map[uuid])
                name = self.id_nodename_map.get(uuid, uuid)
                malicious_names.append(name)
                f.write(f"{uuid}\t{name}\n")

        with open(self._path("malicious.pkl"), "wb") as f:
            pkl.dump((malicious_indices, malicious_names), f)

    def write_preprocess_metadata(self, train_count: int, test_count: int) -> None:
        metadata = {
            "dataset": self.dataset,
            "snapshot": self.snapshot,
            "node_feature_dim": self.node_type_count,
            "edge_feature_dim": self.edge_type_count,
            "n_train": train_count,
            "n_test": test_count,
            "node_type_dict": self.node_type_dict,
            "edge_type_dict": self.edge_type_dict,
        }
        with open(self._path("preprocess_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def run(self) -> None:
        self.ensure_directories()
        self.validate_inputs()
        malicious = self.load_malicious_entities()
        self.collect_entity_metadata()
        self.generate_edge_tables()

        train_graphs = []
        for file_name in DATASET_FILES[self.dataset]["train"]:
            _, graph = self.read_single_graph(file_name, malicious, is_test=False)
            train_graphs.append(graph)

        test_graphs = []
        test_node_maps = []
        for file_name in DATASET_FILES[self.dataset]["test"]:
            node_map, graph = self.read_single_graph(file_name, malicious, is_test=True)
            test_node_maps.append(node_map)
            test_graphs.append(graph)

        self.write_malicious_files(malicious, test_node_maps)
        self.write_snapshots(train_graphs, "train")
        self.write_snapshots(test_graphs, "test")

        print("[5/5] converting snapshots to DGLGraph pickles")
        self.convert_snapshots_to_dgl("train")
        self.convert_snapshots_to_dgl("test")
        self.write_preprocess_metadata(train_count=len(train_graphs), test_count=len(test_graphs))
        print("Done. You can now train with:")
        print(f"  python main.py --dataset {self.dataset} --data-root {self.data_root} --snapshot {self.snapshot} --device auto")


def _graph_dir_sort_key(name: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DARPA TC E3 node-level datasets for CONTINUUM.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES), help="Dataset to preprocess.")
    parser.add_argument("--data-root", default="./datasets", help="Root directory containing dataset folders.")
    parser.add_argument("--snapshot", type=int, default=1, help="Number of timestamp snapshots per raw graph/file.")
    parser.add_argument("--force", action="store_true", help="Regenerate cached intermediate .txt edge tables.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed reserved for future adversarial variants.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = DarpaE3Preprocessor(
        dataset=args.dataset,
        data_root=args.data_root,
        snapshot=args.snapshot,
        force=args.force,
        seed=args.seed,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
