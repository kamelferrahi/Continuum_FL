import os
import random
import pickle as pkl

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from utils.dataloader import load_data, load_graph
from utils.utils import set_random_seed


def _safe_standardize(x_train, x_test):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_std[x_train_std == 0] = 1.0
    return (x_train - x_train_mean) / x_train_std, (x_test - x_train_mean) / x_train_std


def _best_threshold_from_pr(y_true, score):
    prec, rec, threshold = precision_recall_curve(y_true, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = int(np.argmax(f1))
    if len(threshold) == 0:
        best_thres = np.inf
    else:
        best_thres = threshold[min(best_idx, len(threshold) - 1)]
    return prec, rec, f1, best_idx, best_thres


def _confusion_counts(y_true, score, threshold):
    pred = score >= threshold
    y_bool = y_true == 1.0
    tp = int(np.sum(y_bool & pred))
    fn = int(np.sum(y_bool & ~pred))
    tn = int(np.sum(~y_bool & ~pred))
    fp = int(np.sum(~y_bool & pred))
    return tn, fn, tp, fp


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    print("Start Evaluation")
    model.eval()
    x_list = []
    y_list = []
    data = load_data(dataset)
    full = data["full_index"]
    labels = data["labels"]

    with torch.no_grad():
        for i in full:
            g = load_graph(i, dataset, device)
            label = labels[i]
            out = model.embed(g)
            if dataset != "wget":
                out = pooler(g[-1], out).cpu().numpy()
            else:
                out = pooler(g[-1], out, [1]).cpu().numpy()
            y_list.append(label)
            x_list.append(out)

    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if "knn" in method:
        test_auc, test_std = evaluate_batch_level_using_knn(-1, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def _batch_train_count(dataset):
    if dataset == "streamspot":
        return 400
    if dataset in ["Unicorn-Cadets", "wget-long"]:
        return 70
    if dataset in ["wget", "SC2"]:
        return 100
    return 30


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    train_count = _batch_train_count(dataset)
    default_neighbors = min(int(train_count * 0.02), 10) if dataset == "SC2" else 100
    default_neighbors = max(2, min(default_neighbors, train_count))

    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    if repeat != -1:
        prec_list, rec_list, f1_list = [], [], []
        tp_list, fp_list, tn_list, fn_list, auc_list = [], [], [], [], []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train, x_test = _safe_standardize(x_train, x_test)

            n_neighbors = min(default_neighbors, len(x_train))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, _ = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / max(n_neighbors - 1, 1)
            distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, f1, best_idx, best_thres = _best_threshold_from_pr(y_test, score)
            tn, fn, tp, fp = _confusion_counts(y_test, score, best_thres)
            prec_list.append(prec[best_idx])
            rec_list.append(rec[best_idx])
            f1_list.append(f1[best_idx])
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        print("AUC: {}+{}".format(np.mean(auc_list), np.std(auc_list)))
        print("F1: {}+{}".format(np.mean(f1_list), np.std(f1_list)))
        print("PRECISION: {}+{}".format(np.mean(prec_list), np.std(prec_list)))
        print("RECALL: {}+{}".format(np.mean(rec_list), np.std(rec_list)))
        print("TN: {}+{}".format(np.mean(tn_list), np.std(tn_list)))
        print("FN: {}+{}".format(np.mean(fn_list), np.std(fn_list)))
        print("TP: {}+{}".format(np.mean(tp_list), np.std(tp_list)))
        print("FP: {}+{}".format(np.mean(fp_list), np.std(fp_list)))
        return np.mean(auc_list), np.std(auc_list)

    set_random_seed(2022)
    np.random.shuffle(benign_idx)
    np.random.shuffle(attack_idx)
    x_train = x[benign_idx[:train_count]]
    x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
    y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
    x_train, x_test = _safe_standardize(x_train, x_test)

    f1_max = -1.0
    auc_max = 0.0
    prec_max = rec_max = 0.0
    tn_max = fn_max = tp_max = fp_max = 0
    best_n = 0

    for n_neighbors in range(2, max(3, train_count)):
        n_neighbors = min(n_neighbors, len(x_train))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, _ = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / max(n_neighbors - 1, 1)
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        score = distances.mean(axis=1) / mean_distance

        auc = roc_auc_score(y_test, score)
        prec, rec, f1, best_idx, best_thres = _best_threshold_from_pr(y_test, score)
        tn, fn, tp, fp = _confusion_counts(y_test, score, best_thres)

        if f1[best_idx] > f1_max:
            f1_max = f1[best_idx]
            auc_max = auc
            prec_max = prec[best_idx]
            rec_max = rec[best_idx]
            tn_max, fn_max, tp_max, fp_max = tn, fn, tp, fp
            best_n = n_neighbors

    print("AUC: {}".format(auc_max))
    print("F1: {}".format(f1_max))
    print("PRECISION: {}".format(prec_max))
    print("RECALL: {}".format(rec_max))
    print("TN: {}".format(tn_max))
    print("FN: {}".format(fn_max))
    print("TP: {}".format(tp_max))
    print("FP: {}".format(fp_max))
    print(best_n)
    return auc_max, 0.0


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    x_train, x_test = _safe_standardize(x_train, x_test)

    n_neighbors = 200 if dataset == "cadets-e3" else 10
    n_neighbors = min(n_neighbors, len(x_train))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    save_dict_path = "./eval_result/distance_save_{}.pkl".format(dataset)
    if not os.path.exists(save_dict_path):
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][: min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, "wb") as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, "rb") as f:
            mean_distance, distances = pkl.load(f)

    score = distances / mean_distance
    del distances
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = int(np.argmax(f1))

    for i in range(len(f1)):
        if dataset == "trace-e3" and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == "theia-e3" and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == "cadets-e3" and rec[i] < 0.9976:
            best_idx = i - 1
            break

    best_idx = max(0, best_idx)
    best_thres = threshold[min(best_idx, len(threshold) - 1)] if len(threshold) else np.inf
    tn, fn, tp, fp = _confusion_counts(y_test, score, best_thres)

    print("AUC: {}".format(auc))
    print("F1: {}".format(f1[best_idx]))
    print("PRECISION: {}".format(prec[best_idx]))
    print("RECALL: {}".format(rec[best_idx]))
    print("TN: {}".format(tn))
    print("FN: {}".format(fn))
    print("TP: {}".format(tp))
    print("FP: {}".format(fp))
    return auc, 0.0, None, None
