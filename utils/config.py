import argparse


GRAPH_LEVEL_DATASETS = {
    "wget",
    "streamspot",
    "SC2",
    "Unicorn-Cadets",
    "wget-long",
    "clearscope-e3",
}

ENTITY_LEVEL_DATASETS = {
    "cadets-e3",
    "theia-e3",
    "trace-e3",
}

ALL_DATASETS = sorted(GRAPH_LEVEL_DATASETS | ENTITY_LEVEL_DATASETS)


def build_args():
    parser = argparse.ArgumentParser(
        description="CONTINUUM 本地版：不使用 FedML、MPI、MQTT 或同态加密的单机 STGNN Autoencoder 训练与评估。"
    )

    parser.add_argument("--dataset", type=str, default="wget", choices=ALL_DATASETS, help="要运行的数据集名称。")
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录；也可用 CONTINUUM_DATA_ROOT 环境变量设置。")
    parser.add_argument("--snapshot", type=int, default=1, help="每个样本使用的时间快照数量。")
    parser.add_argument("--device", type=str, default="auto", help="auto、cpu、cuda 或 cuda:0 等 torch 设备字符串。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别。")

    parser.add_argument("--epochs", type=int, default=None, help="训练轮数；不指定时按论文实验默认值自动选择。")
    parser.add_argument("--batch-size", type=int, default=1, help="图级任务的 batch size。论文实现要求每批通常为 1 个完整图。")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减。")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "adadelta", "radam", "sgd"], help="优化器。")

    parser.add_argument("--out-dim", type=int, default=64, help="编码器输出维度。")
    parser.add_argument("--gnn-layers", type=int, default=None, help="GAT 层数；不指定时按数据集自动选择。")
    parser.add_argument("--heads", type=int, default=4, help="GAT 多头注意力 head 数。")
    parser.add_argument("--activation", type=str, default="prelu", choices=["relu", "gelu", "prelu", "elu"], help="GNN 激活函数。")
    parser.add_argument("--feat-drop", type=float, default=0.1, help="GAT 特征 dropout。")
    parser.add_argument("--negative-slope", type=float, default=0.2, help="GAT LeakyReLU 负斜率。")
    parser.add_argument("--norm", type=str, default="BatchNorm", help="归一化层名称；保持原实现默认值。")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "sum", "max"], help="图级嵌入聚合方式。")
    parser.add_argument("--loss-fn", type=str, default="sce", choices=["sce", "ce", "mse", "mae"], help="重构损失函数。")
    parser.add_argument("--alpha-l", type=float, default=3, help="SCE loss 的 alpha 指数。")

    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="训练检查点目录。")
    parser.add_argument("--result-dir", type=str, default="./result", help="最终模型目录。")
    parser.add_argument("--eval-result-dir", type=str, default="./eval_result", help="KNN 距离缓存目录。")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定要加载/保存的检查点文件；默认使用 checkpoint-dir 下的 checkpoint-{dataset}.pt。")
    parser.add_argument("--eval-only", action="store_true", help="只评估已有检查点，不重新训练。")
    parser.add_argument("--validation", action="store_true", help="图级训练时每个 epoch 后执行验证。大数据集上会显著变慢。")

    args = parser.parse_args()
    if args.snapshot < 1:
        parser.error("--snapshot 必须大于等于 1。")
    return args
