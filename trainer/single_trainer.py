import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
from utils.poolers import Pooling
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def train_single(main_args, model, dataset):
    device = "cpu"
    set_random_seed(0)
    batch_size = 1
    n_node_feat = dataset['n_feat']
    n_edge_feat = dataset['e_feat']
    graphs = dataset['dataset']
    train_index = dataset['train_index']
    model = model.to(device)
    optimizer = create_optimizer(main_args["optimizer"], model, main_args["lr"], main_args["weight_decay"])
    model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args["max_epoch"], device, main_args["n_dim"], main_args["e_dim"])
    #torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format("wget"))
    pooler = Pooling(main_args["pooling"])
    test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], "wget" ,main_args["n_dim"],  main_args["e_dim"])
    return test_auc, model



