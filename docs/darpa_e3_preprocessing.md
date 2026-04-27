# DARPA E3 节点级数据预处理说明

本文档说明如何把 DARPA Transparent Computing E3 原始 CDM JSON 数据转换为本项目 `main.py` 可以直接读取的 DGLGraph 快照数据。

当前支持的数据集：

- `theia-e3`
- `trace-e3`
- `cadets-e3`

预处理入口：

```bash
python -m preprocess.darpa_e3 --dataset theia-e3 --data-root ./datasets --snapshot 1
```

---

## 1. 这个预处理做了什么

论文第 5.2 节的数据构造流程包括：动态快照生成、节点和边类型编码、边压缩、转换为 DGL graph。本项目中的 `preprocess/darpa_e3.py` 将这些步骤落地为如下流水线：

```text
DARPA E3 原始 CDM JSON
    ↓
扫描实体 UUID、实体类型、实体名称
    ↓
解析 Event 记录，抽取 source / destination / edge type / timestamp
    ↓
生成中间边表 *.json.txt
    ↓
构建 NetworkX MultiDiGraph provenance graph
    ↓
按 timestamp 切分为 snapshot
    ↓
在每个 snapshot 内压缩重复边
    ↓
转换为 DGLGraph
    ↓
保存为 train/graph*/graph*.pkl 和 test/graph*/graph*.pkl
```

最终生成的数据可以直接用于：

```bash
python main.py --dataset theia-e3 --data-root ./datasets --snapshot 1 --device auto
```

---

## 2. 输入文件要求

`--data-root` 指向所有数据集的根目录。以默认 `./datasets` 为例，`theia-e3` 的原始数据应放在：

```text
./datasets/theia-e3/
```

### 2.1 先理解“哪些文件用于训练/测试”与“哪些文件建议保留”的区别

DARPA E3 原始数据通常会被拆成多个连续分片，例如：

```text
ta1-theia-e3-official-6r.json
ta1-theia-e3-official-6r.json.1
ta1-theia-e3-official-6r.json.2
ta1-theia-e3-official-6r.json.3
ta1-theia-e3-official-6r.json.4
ta1-theia-e3-official-6r.json.5
ta1-theia-e3-official-6r.json.6
ta1-theia-e3-official-6r.json.7
ta1-theia-e3-official-6r.json.8
```

原始 CONTINUUM 实验脚本并不是把所有分片都作为训练/测试样本。它对 `theia-e3` 采用了一个硬编码划分：

- `ta1-theia-e3-official-6r.json`、`.1`、`.2`、`.3` 用作训练来源。
- `ta1-theia-e3-official-6r.json.8` 用作测试来源。

所以文档中的 `json、json.1、json.2、json.3、json.8` 指的是**当前复现实验划分真正参与构图、切快照、训练和测试的文件**，不是说原始压缩包里只应该有这几个文件。

如果你手里有 `.4`、`.5`、`.6`、`.7`，建议也保留在 `datasets/theia-e3/` 目录中，不要删掉。它们目前不会被当作训练/测试样本使用，但在一些原始数据发布和解析场景中，额外分片可能包含后续事件引用到的实体定义或辅助记录。保留它们更安全，也方便后续扩展实验划分。

---

### 2.2 theia-e3：最小必需结构

如果只想按当前 CONTINUUM 实验划分运行，最小必需文件是：

```text
datasets/
└── theia-e3/
    ├── ta1-theia-e3-official-6r.json
    ├── ta1-theia-e3-official-6r.json.1
    ├── ta1-theia-e3-official-6r.json.2
    ├── ta1-theia-e3-official-6r.json.3
    ├── ta1-theia-e3-official-6r.json.8
    └── theia-e3.txt
```

其中：

- 前 4 个 json 文件作为训练来源。
- `ta1-theia-e3-official-6r.json.8` 作为测试来源。
- `theia-e3.txt` 是恶意实体 UUID 列表，一行一个 UUID，用于生成 `malicious.pkl`。

### 2.3 theia-e3：推荐保留结构

如果你从原始压缩包解压后得到了更多连续分片，推荐保留为：

```text
datasets/
└── theia-e3/
    ├── ta1-theia-e3-official-6r.json
    ├── ta1-theia-e3-official-6r.json.1
    ├── ta1-theia-e3-official-6r.json.2
    ├── ta1-theia-e3-official-6r.json.3
    ├── ta1-theia-e3-official-6r.json.4
    ├── ta1-theia-e3-official-6r.json.5
    ├── ta1-theia-e3-official-6r.json.6
    ├── ta1-theia-e3-official-6r.json.7
    ├── ta1-theia-e3-official-6r.json.8
    └── theia-e3.txt
```

当前默认脚本仍只会把 `json、json.1、json.2、json.3` 作为训练，把 `json.8` 作为测试。

---

### 2.4 trace-e3

```text
datasets/
└── trace-e3/
    ├── ta1-trace-e3-official-1.json
    ├── ta1-trace-e3-official-1.json.1
    ├── ta1-trace-e3-official-1.json.2
    ├── ta1-trace-e3-official-1.json.3
    ├── ta1-trace-e3-official-1.json.4
    └── trace-e3.txt
```

当前默认划分中，`json`、`.1`、`.2`、`.3` 用作训练来源；`json`、`.1`、`.2`、`.3`、`.4` 用作测试来源。

### 2.5 cadets-e3

```text
datasets/
└── cadets-e3/
    ├── ta1-cadets-e3-official.json
    ├── ta1-cadets-e3-official.json.1
    ├── ta1-cadets-e3-official.json.2
    ├── ta1-cadets-e3-official-2.json
    ├── ta1-cadets-e3-official-2.json.1
    └── cadets-e3.txt
```

当前默认划分中，`ta1-cadets-e3-official.json`、`.1`、`.2`、`ta1-cadets-e3-official-2.json.1` 用作训练来源；`ta1-cadets-e3-official-2.json` 用作测试来源。

---

## 3. 运行预处理

### 3.1 预处理 theia-e3

```bash
python -m preprocess.darpa_e3 \
  --dataset theia-e3 \
  --data-root ./datasets \
  --snapshot 1
```

### 3.2 预处理 trace-e3

```bash
python -m preprocess.darpa_e3 \
  --dataset trace-e3 \
  --data-root ./datasets \
  --snapshot 1
```

### 3.3 预处理 cadets-e3

```bash
python -m preprocess.darpa_e3 \
  --dataset cadets-e3 \
  --data-root ./datasets \
  --snapshot 1
```

### 3.4 重新生成中间边表

预处理会缓存中间边表，例如：

```text
ta1-theia-e3-official-6r.json.txt
```

如果原始 json 或恶意实体文件发生变化，建议加 `--force`：

```bash
python -m preprocess.darpa_e3 \
  --dataset theia-e3 \
  --data-root ./datasets \
  --snapshot 1 \
  --force
```

---

## 4. 输出文件结构

以 `theia-e3`、`--snapshot 1` 为例，预处理后会得到：

```text
datasets/
└── theia-e3/
    ├── ta1-theia-e3-official-6r.json
    ├── ta1-theia-e3-official-6r.json.1
    ├── ta1-theia-e3-official-6r.json.2
    ├── ta1-theia-e3-official-6r.json.3
    ├── ta1-theia-e3-official-6r.json.8
    ├── theia-e3.txt
    ├── ta1-theia-e3-official-6r.json.txt
    ├── ta1-theia-e3-official-6r.json.1.txt
    ├── ta1-theia-e3-official-6r.json.2.txt
    ├── ta1-theia-e3-official-6r.json.3.txt
    ├── ta1-theia-e3-official-6r.json.8.txt
    ├── names.json
    ├── types.json
    ├── malicious.pkl
    ├── malicious_names.txt
    ├── preprocess_metadata.json
    ├── train/
    │   ├── graph0/
    │   │   └── graph0.pkl
    │   ├── graph1/
    │   │   └── graph0.pkl
    │   ├── graph2/
    │   │   └── graph0.pkl
    │   └── graph3/
    │       └── graph0.pkl
    └── test/
        └── graph0/
            └── graph0.pkl
```

如果 `--snapshot 3`，每个 `graphN/` 目录下会生成：

```text
graph0.pkl
graph1.pkl
graph2.pkl
```

含义是：

```text
graphN/      第 N 个原始日志文件构成的样本
graphK.pkl   第 K 个 timestamp 时间快照
```

---

## 5. 预处理后训练

预处理完成后，直接运行：

```bash
python main.py \
  --dataset theia-e3 \
  --data-root ./datasets \
  --snapshot 1 \
  --device auto
```

如果预处理时用了 `--snapshot 3`，训练时也必须使用同样的 snapshot 数：

```bash
python main.py \
  --dataset theia-e3 \
  --data-root ./datasets \
  --snapshot 3 \
  --device auto
```

---

## 6. `theia-e3.txt` / `trace-e3.txt` / `cadets-e3.txt` 是什么

这些文件是恶意实体 UUID 列表，用来生成测试标签 `malicious.pkl`。

格式：

```text
uuid-1
uuid-2
uuid-3
```

每行一个 UUID。预处理脚本会把这些 UUID 映射到测试图中的节点编号，并过滤掉 `MemoryObject` 和 `UnnamedPipeObject` 等不作为最终检测目标的实体。

如果缺少这个文件，脚本会停止并提示缺失文件。

---

## 7. 与论文流程的对应关系

| 论文中的步骤 | 本项目实现位置 |
| --- | --- |
| 解析原始 provenance 日志 | `collect_entity_metadata()`、`generate_edge_tables()` |
| 提取 source/destination/type/timestamp | `generate_edge_tables()` |
| 构建 provenance graph | `read_single_graph()` |
| 动态快照生成 | `write_snapshots()` |
| 节点类型编码 | `convert_snapshots_to_dgl()` 中的 `one_hot()` |
| 边类型编码 | `write_snapshots()` 中的 `attr` 向量 |
| 边压缩 | `write_snapshots()` 中的 `has_edge()` + 计数累加 |
| 转换 DGLGraph | `convert_snapshots_to_dgl()` |
| 保存 PKL | `write_snapshots()` 和 `convert_snapshots_to_dgl()` |

---

## 8. 注意事项

1. DARPA E3 原始 JSON 很大，预处理会消耗较多内存和磁盘空间。
2. `--snapshot` 越大，输出文件越多，训练时需要读取更多快照。
3. 如果你重新运行预处理，建议先备份或清理旧的 `train/`、`test/` 输出目录，避免混合旧快照和新快照。
4. 当前预处理仅覆盖节点级 E3 数据集，不覆盖 `wget`、`streamspot`、`SC2` 等图级数据集。
5. 预处理完成后首次训练时，如果没有 `metadata.json`，`utils/dataloader.py` 会根据 `malicious.pkl` 和 `train/graph0/graph0.pkl` 自动生成。
