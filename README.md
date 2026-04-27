# CONTINUUM 本地版（移除联邦学习）

本分支将 CONTINUUM 项目改为**单机/单进程训练与评估版本**。代码保留论文中的核心检测模型：空间-时间图神经网络自编码器（STGNN Autoencoder）+ KNN 异常检测；同时移除 FedML、MPI、MQTT、客户端/服务端聚合器、联邦学习配置和同态加密通信入口。

> 说明：论文中的联邦学习与同态加密属于部署层设计，用于多客户端协同训练和隐私保护。本地版不再进行跨客户端权重聚合，只在一台机器上读取预处理后的图快照进行训练与评估。

---

## 1. 本分支做了什么

- `main.py`：重写为本地训练入口，不再调用 `fedml.init()`、`FedMLRunner`、客户端训练器或服务端聚合器。
- `utils/config.py`：改为普通命令行参数解析，不再读取 `fedml_config.yaml`。
- `utils/dataloader.py`：删除联邦学习数据分片函数，保留图级和实体级数据加载；支持 `--data-root` 或 `CONTINUUM_DATA_ROOT` 指定数据集根目录。
- `requirement.txt`：删除 FedML、MPI、MQTT、MLOps 等联邦部署依赖，仅保留本地训练所需核心包。
- 删除联邦学习相关文件：`fedml_config.yaml`、`gpu_mapping.yaml`、`mpi_host_file`、`trainer/magic_trainer.py`、`trainer/magic_aggregator.py`。

---

## 2. 方法流程

本地版的检测流程如下：

1. 读取已经预处理好的 provenance graph 快照。
2. 使用带边特征的 GAT 捕获单个快照内的空间依赖关系。
3. 使用 GRU 追踪多个快照之间的时间演化。
4. 通过 STGNN Autoencoder 重构节点特征，并学习节点/图嵌入。
5. 使用 KNN 在良性样本嵌入上建立正常行为参考；距离异常的节点或图被判定为可疑。

---

## 3. 环境安装

推荐使用 Conda 创建独立环境：

```bash
conda create -n continuum-local python=3.9 -y
conda activate continuum-local
pip install -r requirement.txt
```

如果需要 GPU，请先根据本机 CUDA 版本安装 PyTorch。例如 CUDA 12.1：

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html
pip install -r requirement.txt
```

如果没有 GPU，运行时使用 `--device cpu`。

---

## 4. 数据集目录

代码默认从 `./datasets` 读取数据。也可以通过环境变量或命令行参数指定：

```bash
export CONTINUUM_DATA_ROOT=/path/to/datasets
```

或：

```bash
python main.py --dataset wget --data-root /path/to/datasets
```

`--data-root` 应指向所有数据集的上级目录，而不是某个具体数据集目录。

### 4.1 支持的数据集名称

图级检测：

- `wget`
- `wget-long`
- `streamspot`
- `SC2`
- `Unicorn-Cadets`
- `clearscope-e3`

实体级/节点级检测：

- `cadets-e3`
- `theia-e3`
- `trace-e3`

### 4.2 图级数据示例

`wget` 和 `streamspot`：

```text
/path/to/datasets/
└── wget/
    └── cache/
        ├── graph0/
        │   ├── graph0.pkl
        │   ├── graph1.pkl
        │   └── ...
        └── graph1/
```

`SC2`、`Unicorn-Cadets`、`wget-long`、`clearscope-e3`：

```text
/path/to/datasets/
└── SC2/
    ├── node.pkl
    ├── edge.pkl
    └── cache/
        ├── benign/
        │   └── graph0/
        │       ├── graph0.pkl
        │       └── ...
        └── attack/
            └── graph0/
                ├── graph0.pkl
                └── ...
```

### 4.3 实体级数据示例

```text
/path/to/datasets/
└── trace-e3/
    ├── malicious.pkl
    ├── metadata.json          # 可选；没有时会自动生成
    ├── train/
    │   └── graph0/
    │       ├── graph0.pkl
    │       └── ...
    └── test/
        └── graph0/
            ├── graph0.pkl
            └── ...
```

每个 `graph*.pkl` 应为 DGLGraph，并包含：

```python
g.ndata["attr"]  # 节点特征
g.edata["attr"]  # 边特征
```

图级 pooling 如需按节点类型筛选，还应包含：

```python
g.ndata["type"]
```

---

## 5. 运行方法：训练和评估的区别

`main.py` 有两种运行模式：

| 模式 | 是否训练模型 | 是否评估模型 | 何时使用 |
| --- | --- | --- | --- |
| **训练 + 自动评估** | 是 | 是 | 第一次运行某个数据集，或想重新训练模型。 |
| **只评估已有检查点** | 否 | 是 | 已经有训练好的 `checkpoint-*.pt`，只想复现实验指标或快速测试。 |

也就是说：

- **不加 `--eval-only`**：先训练 STGNN Autoencoder，然后自动用 KNN 做评估。
- **加 `--eval-only`**：跳过训练，只加载 `--checkpoint` 指定的已有模型，然后做 KNN 评估。
- 图级检测和实体级/节点级检测都遵循这个规则；差别只在于 `--dataset` 选择不同。
- `--validation` 不是最终评估。它只是在图级训练过程中每个 epoch 后额外跑一次验证，默认关闭，因为会明显变慢。

### 5.1 图级检测：训练 + 自动评估

下面命令会对 `wget` 数据集执行完整流程：

1. 读取训练图。
2. 训练 STGNN Autoencoder。
3. 保存检查点到 `checkpoints/checkpoint-wget.pt`。
4. 自动在图级嵌入上运行 KNN 异常检测并打印 AUC、F1、Precision、Recall 等指标。
5. 保存最终模型到 `result/local-wget.pt`。

```bash
python main.py \
  --dataset wget \
  --data-root /path/to/datasets \
  --snapshot 1 \
  --device auto
```

### 5.2 实体级/节点级检测：训练 + 自动评估

下面命令会对 `trace-e3` 数据集执行完整流程：

1. 读取训练图快照。
2. 训练 STGNN Autoencoder。
3. 保存检查点到 `checkpoints/checkpoint-trace-e3.pt`。
4. 生成训练集和测试集节点嵌入。
5. 用 KNN 做节点级异常检测并打印指标。
6. 保存最终模型到 `result/local-trace-e3.pt`。

```bash
python main.py \
  --dataset trace-e3 \
  --data-root /path/to/datasets \
  --snapshot 1 \
  --device cuda:0
```

### 5.3 图级检测：只评估已有检查点

下面命令不会训练模型。它会直接加载 `checkpoints/checkpoint-wget.pt`，然后对 `wget` 做图级 KNN 评估。

```bash
python main.py \
  --dataset wget \
  --data-root /path/to/datasets \
  --snapshot 1 \
  --device cpu \
  --eval-only \
  --checkpoint checkpoints/checkpoint-wget.pt
```

### 5.4 实体级/节点级检测：只评估已有检查点

下面命令不会训练模型。它会直接加载 `checkpoints/checkpoint-trace-e3.pt`，然后对 `trace-e3` 做节点级 KNN 评估。

```bash
python main.py \
  --dataset trace-e3 \
  --data-root /path/to/datasets \
  --snapshot 1 \
  --device cuda:0 \
  --eval-only \
  --checkpoint checkpoints/checkpoint-trace-e3.pt
```

---

## 6. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `wget` | 数据集名称。决定运行图级检测还是实体级/节点级检测。 |
| `--data-root` | `./datasets` | 数据集根目录。 |
| `--snapshot` | `1` | 每个样本读取的快照数量。 |
| `--device` | `auto` | `auto`、`cpu`、`cuda`、`cuda:0` 等。 |
| `--epochs` | 自动 | 训练轮数。图级默认 6，实体级默认 50。只评估时不会使用。 |
| `--batch-size` | `1` | 图级训练 batch size。只评估时不会使用。 |
| `--lr` | `0.0001` | 学习率。只评估时不会使用。 |
| `--optimizer` | `adam` | 训练优化器，可选 `adam`、`adamw`、`adadelta`、`radam`、`sgd`。只评估时不会使用。 |
| `--out-dim` | `64` | STGNN 隐层/输出维度。训练和加载检查点时需保持一致。 |
| `--gnn-layers` | 自动 | GAT 层数。训练和加载检查点时需保持一致。 |
| `--heads` | `4` | GAT 注意力头数。训练和加载检查点时需保持一致。 |
| `--pooling` | `mean` | 图级嵌入聚合方式。 |
| `--checkpoint-dir` | `./checkpoints` | 训练后保存检查点的目录。 |
| `--result-dir` | `./result` | 最终模型保存目录。 |
| `--checkpoint` | 自动 | 只评估或加载指定模型时使用的检查点路径。 |
| `--eval-only` | 关闭 | 开启后跳过训练，只加载检查点并评估。 |
| `--validation` | 关闭 | 图级训练过程中的中间验证，不是最终评估；默认关闭。 |

---

## 7. 输出文件

执行“训练 + 自动评估”时会生成：

```text
checkpoints/checkpoint-{dataset}.pt
result/local-{dataset}.pt
eval_result/distance_save_{dataset}.pkl
```

执行“只评估已有检查点”时不会重新训练，也不会覆盖 `checkpoints/checkpoint-{dataset}.pt`；但仍可能生成或复用 `eval_result/distance_save_{dataset}.pkl` 作为 KNN 距离缓存。

终端会打印 AUC、F1、Precision、Recall、TP、FP、TN、FN 等指标。

---

## 8. 与旧版联邦学习运行方式的区别

旧版：

```bash
mpirun -np 4 --hostfile mpi_host_file --oversubscribe python main.py --cf fedml_config.yaml
```

本地版：

```bash
python main.py --dataset wget --data-root /path/to/datasets --snapshot 1 --device auto
```

本地版不再需要客户端数量、聚合轮次、`FedAvg`/`FedProx`/`FedOpt`、MQTT/S3、GPU mapping 或同态加密通信配置。

---

## 9. 常见问题

### 找不到数据集

请确认 `--data-root` 指向数据集根目录。例如：

```bash
python main.py --dataset wget --data-root /path/to/datasets
```

而不是：

```bash
python main.py --dataset wget --data-root /path/to/datasets/wget
```

### CUDA 不可用

改用 CPU：

```bash
python main.py --dataset wget --device cpu
```

### 内存或显存不足

APT provenance graph 可能很大。可以先用 `wget` 或 `streamspot` 验证流程，关闭 `--validation`，减少 `--snapshot`，或改用 CPU。
