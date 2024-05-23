import logging
import os
import random
import torch
import warnings
from tqdm import tqdm
import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from fedml.core import ClientTrainer
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

from model.train_graph import batch_level_train
from model.train_entity import entity_level_train
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn


from utils.utils import set_random_seed, create_optimizer
from utils.poolers import Pooling
from utils.config import build_args
from utils.dataloader import load_data, load_entity_level_dataset, load_metadata

# Trainer for MoleculeNet. The evaluation metric is ROC-AUC
def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader



class MagicTrainer(ClientTrainer):
    def __init__(self, model, args, name):
        super().__init__(model, args)
        self.name = name
        self.max = 0
    	
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        main_args = build_args()
        dataset_name = self.name
        input('start')
        if (dataset_name in ['wget', 'streamspot', 'SC2', 'Unicorn-Cadets', 'wget-long', 'clearscope-e3']):
            if (dataset_name == 'wget' or dataset_name == 'streamspot' or dataset_name == 'Unicorn-Cadets'  or dataset_name == 'clearscope-e3'):
                batch_size = 1
                main_args.max_epoch = 6

            elif (dataset_name == 'SC2' or dataset_name == 'wget-long'):
                batch_size = 1
                main_args.max_epoch = 1
 
        
            dataset = load_data(dataset_name)
            n_node_feat = dataset['n_feat']
            n_edge_feat = dataset['e_feat']
            #train_index = [104, 118, 86, 74, 16, 12, 117, 108, 59, 146, 97, 49, 107, 47, 23, 111, 32, 124, 121, 119, 141, 50, 43, 98, 73, 80, 4, 140, 1, 17, 55, 136, 95, 120, 103, 94, 34, 68, 130, 26, 30, 29, 129, 71, 6, 128, 84, 85, 72, 96, 87, 58, 81, 79, 31, 37, 54, 93, 135, 33, 61, 134, 52, 106, 126, 139, 8, 115, 82, 46, 101, 114, 60, 138, 132, 5, 2, 19, 143, 77, 92, 123, 42, 113, 125, 15, 105, 14, 145, 148]
            validation_index = dataset['validation_index']
            label = dataset["labels"]
            main_args.n_dim = n_node_feat
            main_args.e_dim = n_edge_feat
            main_args.optimizer = "adamw"
            set_random_seed(0)
            #model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
            optimizer = create_optimizer(main_args.optimizer, self.model, main_args.lr, main_args.weight_decay)
            train_loader = extract_dataloaders(train_data[0], batch_size)
            self.model = batch_level_train(self.model,  train_loader, optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim, dataset_name, validation= False)
        else:
            main_args.max_epoch = 50            
            nsnapshot = args.snapshot
            main_args.optimizer = "adam"
            set_random_seed(0)
            #model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
            optimizer = create_optimizer(main_args.optimizer, self.model, main_args.lr, main_args.weight_decay)
            #train_loader = extract_dataloaders(train_index, batch_size)
            self.model = entity_level_train(self.model, nsnapshot, optimizer, main_args.max_epoch, device, dataset_name, train_data[0])

        self.max = 0
        best_model_params = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
            
     

        return self.max, best_model_params

    def test(self, test_data, device, args):
        if (self.name == 'wget' or self.name == 'streamspot'):
            logging.info("----------test--------")
            main_args = build_args()        
            dataset_name = self.name
            if dataset_name in ['streamspot', 'wget', 'SC2']:
                main_args.num_hidden = 256
                main_args.num_layers = 4
                main_args.max_epoch = 50
            else:
                main_args.num_hidden = 64
                main_args.num_layers = 3
            set_random_seed(0)
            dataset = load_data(dataset_name, 1, 0.6, 0.2)
            n_node_feat = dataset['n_feat']
            n_edge_feat = dataset['e_feat']
                #train_index = [104, 118, 86, 74, 16, 12, 117, 108, 59, 146, 97, 49, 107, 47, 23, 111, 32, 124, 121, 119, 141, 50, 43, 98, 73, 80, 4, 140, 1, 17, 55, 136, 95, 120, 103, 94, 34, 68, 130, 26, 30, 29, 129, 71, 6, 128, 84, 85, 72, 96, 87, 58, 81, 79, 31, 37, 54, 93, 135, 33, 61, 134, 52, 106, 126, 139, 8, 115, 82, 46, 101, 114, 60, 138, 132, 5, 2, 19, 143, 77, 92, 123, 42, 113, 125, 15, 105, 14, 145, 148]
            main_args.n_dim = n_node_feat
            main_args.e_dim = n_edge_feat
            model = self.model
            model.eval()
            model.to(device)
            model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
            model = model.to(device)
            pooler = Pooling(main_args.pooling)
            test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], dataset_name, main_args.n_dim,
                                                                    main_args.e_dim)
            
        else:
            metadata = load_metadata(self.name)
            args["n_dim"] = metadata['node_feature_dim']
            args["e_dim"] = metadata['edge_feature_dim']
            model = self.model.to(device)
            model.eval()
            malicious, _ = metadata['malicious']
            n_train = metadata['n_train']
            n_test = metadata['n_test']
            with torch.no_grad():
                x_train = []
                for i in range(n_train):
                   g = load_entity_level_dataset(self.name, 'train', i).to(device)
                   x_train.append(model.embed(g).cpu().detach().numpy())
                   del g
                x_train = np.concatenate(x_train, axis=0)
                skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(self.name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().detach().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(self.name, x_train, result_x_test,
                                                                       result_y_test)

        return test_auc, model

