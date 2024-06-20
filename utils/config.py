import argparse


def build_args():
    parser = argparse.ArgumentParser(description="MAGIC")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_fn", type=str, default='sce')
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--run_id", type=str, default="fedgraphnn_cs_ch")
    parser.add_argument("--cf", type=str, default="fedml_config.yaml")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--role", type=str, default="server")
    args = parser.parse_args()
    return args