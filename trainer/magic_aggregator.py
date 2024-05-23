import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils.config import build_args
from fedml.core import ServerAggregator
from model.train_graph import batch_level_train
from model.train_entity import entity_level_train
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn


from utils.utils import set_random_seed, create_optimizer
from utils.poolers import Pooling
from utils.config import build_args
from utils.dataloader import load_data, load_entity_level_dataset, load_metadata

class MagicWgetAggregator(ServerAggregator):
    def __init__(self, model, args, name):
        super().__init__(model, args)
        self.name = name
        
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self._test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info("Client {}, Test ROC-AUC score = {}".format(client_idx, score))
            if args.enable_wandb:
                wandb.log({"Client {} Test/ROC-AUC".format(client_idx): score})
        avg_score = np.mean(np.array(score_list))
        logging.info("Test ROC-AUC Score = {}".format(avg_score))
        if args.enable_wandb:
            wandb.log({"Test/ROC-AUC": avg_score})
        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")

    def _test(self, test_data, device, args):
        main_args = build_args()        
        dataset_name = self.name
        nsnapshot = args.snapshot
        if (self.name in ['wget', 'streamspot', 'SC2', 'Unicorn-Cadets', 'wget-long', 'clearscope-e3']):
            logging.info("----------test--------") 
            set_random_seed(0)
            dataset = load_data(dataset_name)
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
            main_args.n_dim = metadata['node_feature_dim']
            main_args.e_dim = metadata['edge_feature_dim']
            model = self.model.to(device)
            model.eval()
            malicious, _ = metadata['malicious']
            n_train = metadata['n_train']
            n_test = metadata['n_test']

            with torch.no_grad():
                x_train = []
                for i in range(n_train):
                   g = load_entity_level_dataset(dataset_name, 'train', i, nsnapshot, device)
                   x_train.append(model.embed(g).cpu().detach().numpy())
                   del g
                x_train = np.concatenate(x_train, axis=0)
                skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(self.name, 'test', i, nsnapshot, device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g[0].number_of_nodes()
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
        torch.save(model.state_dict(), "./result/FedAvg-{}.pt".format(self.name))
        return test_auc, model
                                                    
