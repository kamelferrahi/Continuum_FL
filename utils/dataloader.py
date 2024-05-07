import os
import random
import networkx as nx
import dgl
import torch
import pickle as pkl
import json
import logging
import numpy as np

path_dataset = 'D:/PFE DATASETS/'

def darpa_split(name):
    metadata = load_metadata(name)
    n_train = metadata['n_train']
    train_dataset = range(n_train)
    train_labels = [0]* n_train
    
    
    return (
        train_dataset,
        train_labels,
        [],
        [],
        [],
        []
    )
       

def create_random_split(name, snapshots):
    dataset = load_data(name, 0.6, 0.2 , 0.2)
    # Random 80/10/10 split as suggested 
    

    all_idxs = list(range(len(dataset)))
    random.shuffle(all_idxs)

    train_dataset = dataset['train_index']
    train_labels = []
    for id in train_dataset:
        train_labels.append(dataset['labels'][id])

    val_dataset = dataset['validation_index']
    val_labels = []
    for id in val_dataset:
        val_labels.append(dataset['labels'][id])

    test_dataset = dataset['test_index']
    test_labels = []
    for id in test_dataset:
        test_labels.append(dataset['labels'][id])


    return (
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        test_dataset,
        test_labels,
    )



def partition_data_by_sample_size(
     client_number, name, snapshots
):
    if (name in ['wget', 'streamspot', 'SC2', 'Unicorn-Cadets', 'wget-long', 'clearscope-e3']):
        (
            train_dataset,
            train_labels,
            val_dataset,
            val_labels,
            test_dataset,
            test_labels,
        ) = create_random_split(name, snapshots)
    else:
        (
            train_dataset,
            train_labels,
            val_dataset,
            val_labels,
            test_dataset,
            test_labels,
        ) = darpa_split(name)
        
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    train_idxs = list(range(num_train_samples))
    val_idxs = list(range(num_val_samples))
    test_idxs = list(range(num_test_samples))

    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    partition_dicts = [None] * client_number

    
    clients_idxs_train = np.array_split(train_idxs, client_number)
    clients_idxs_val = np.array_split(val_idxs, client_number)
    clients_idxs_test = np.array_split(test_idxs, client_number)
    
    labels_of_all_clients = []
    for client in range(client_number):
        client_train_idxs = clients_idxs_train[client]
        client_val_idxs = clients_idxs_val[client]
        client_test_idxs = clients_idxs_test[client]

        train_dataset_client = [
            train_dataset[idx] for idx in client_train_idxs
        ]
        train_labels_client = [train_labels[idx] for idx in client_train_idxs]
        labels_of_all_clients.append(train_labels_client)

        val_dataset_client = [val_dataset[idx] for idx in client_val_idxs]
        val_labels_client = [val_labels[idx] for idx in client_val_idxs]

        test_dataset_client = [test_dataset[idx] for idx in client_test_idxs]
        test_labels_client = [test_labels[idx] for idx in client_test_idxs]


        partition_dict = {
            "train": train_dataset_client,
            "val": val_dataset_client,
            "test": test_dataset_client,
        }

        partition_dicts[client] = partition_dict
    global_data_dict = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    return global_data_dict, partition_dicts

def load_partition_data(
    client_number,
    name,
    snapshots,
    global_test=True,
):
    global_data_dict, partition_dicts = partition_data_by_sample_size(
        client_number, name, snapshots
    )

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

  

    # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
    train_data_global =  global_data_dict["train"]
    val_data_global = global_data_dict["val"]
    test_data_global = global_data_dict["test"]
    train_data_num = len(global_data_dict["train"])
    val_data_num = len(global_data_dict["val"])
    test_data_num = len(global_data_dict["test"])

    for client in range(client_number):
        train_dataset_client = partition_dicts[client]["train"]
        val_dataset_client = partition_dicts[client]["val"]
        test_dataset_client = partition_dicts[client]["test"]

        data_local_num_dict[client] = len(train_dataset_client)
        train_data_local_dict[client] = train_dataset_client,
 
        val_data_local_dict[client] = val_dataset_client
  
        test_data_local_dict[client] = (
            test_data_global
            if global_test
            else test_dataset_client
                
        )

        logging.info(
            "Client idx = {}, local sample number = {}".format(
                client, len(train_dataset_client)
            )
        )

    return (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    )







def preload_entity_level_dataset(name):
    path = path_dataset + name
    if os.path.exists(path + '/metadata.json'):
        pass
    else:
    
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))

        n_train = len(os.listdir(path + '/train'))
        n_test = len(os.listdir(path + '/test'))
            
        g = pkl.load(open(path + '/train/graph0/graph0.pkl', 'rb'))

        node_feature_dim = len(g.ndata['attr'][0])
        edge_feature_dim = len(g.edata['attr'][0])

        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': n_train,
            'n_test': n_test
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)



def load_metadata(name):
    preload_entity_level_dataset(name)
    with open( path_dataset + name + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(name, t, n, snapshot, device):
    preload_entity_level_dataset(name)
    graphs = []
    for i in range(snapshot):
        with open(path_dataset + name + '/' + t + '/graph{}/graph{}.pkl'.format(n, str(i)), 'rb') as f:
            graphs.append(pkl.load(f).to(device))
    return graphs


def get_labels(name):
    if (name=="wget" ):
        return [1] * 25 + [0] * 125
    elif (name=="streamspot"):
        return [0] * 300 + [1] * 100 + [0] * 200
    elif (name == 'SC2'):
        return [0] * 125 + [1] * 25 
    elif (name == 'Unicorn-Cadets'):
        return [0] * 109 + [1] * 3
    elif (name == 'wget-long'):
        return [0] * 100 + [1] * 5
    elif (name == 'clearscope-e3'):
        return [0] * 44 + [1] * 50
    
def load_data(name, nsnapshot, train_percent, validation_percent):
    #dataset, n_dim, e_dim = load_raw_data(name, nsnapshot)
    if (name == "wget"):
        n = 150
        n_dim = 14
        e_dim = 4
        n_train = int(n * train_percent)
        n_validation = int(n * validation_percent)
        full_dataset_index =  list(range(n))
        random.shuffle(full_dataset_index)
        #train_dataset = full_dataset_index[:n_train]
        train_dataset = list(range(50, 150))
        #validation_dataset = full_dataset_index[n_train: n_train + n_validation]
        #test_dataset = full_dataset_index[ n_train + n_validation:]
        validation_dataset =  list(range(50))
        test_dataset = list(range(50))

    elif (name == "streamspot"):

        n = 600
        n_train = int(n * train_percent)
        n_validation = int(n * validation_percent)
        full_dataset_index =  list(range(n))
        random.shuffle(full_dataset_index)
        train_dataset = list(range(300)) 
        validation_dataset =  list(range(300, 350)) + list(range(500,550))
        test_dataset = list(range(300, 400)) + list(range(400,500))+ list(range(500,600))

    elif (name == 'SC2'):

        n = 150
        n_dim = len(pkl.load(open(path_dataset + 'SC2/node.pkl', 'rb')).keys())
        e_dim = len(pkl.load(open(path_dataset + 'SC2/edge.pkl', 'rb')).keys())
        full_dataset_index=  list(range(n))
        train_dataset = list(range(0, 100))
        #validation_dataset = full_dataset_index[n_train: n_train + n_validation]
        #test_dataset = full_dataset_index[ n_train + n_validation:]
        validation_dataset =  list(range(0, 150))
        test_dataset = list(range(0, 150))

    elif(name == 'Unicorn-Cadets' or name == 'wget-long' or name == 'clearscope-e3'):
        n_dim = len(pkl.load(open(path_dataset + '{}/node.pkl'.format(name), 'rb')).keys())
        e_dim = len(pkl.load(open(path_dataset + '{}/edge.pkl'.format(name), 'rb')).keys())
        
        if (name == 'Unicorn-Cadets'):
            n = 112
            train_dataset = list(range(0, 70))
            validation_dataset =  list(range(70, 112))
            test_dataset = list(range(70, 112))
        elif (name == 'wget-long'):
            n = 105 
            train_dataset = list(range(0, 70))
            validation_dataset =  list(range(70, 105))
            test_dataset = list(range(70, 105))

        else:
            n = 94
            train_dataset = list(range(0, 30))
            validation_dataset =  list(range(30, 94))
            test_dataset = list(range(30, 94))

        full_dataset_index=  list(range(n))



    return {'dataset': full_dataset_index,
            'train_index': train_dataset,
            'test_index': test_dataset,
            'validation_index': validation_dataset,
            'full_index': full_dataset_index,
            'n_feat': n_dim,
            'e_feat': e_dim,
            'labels': get_labels(name)}

            
def load_graph(id, name ,device):
    graphs = []

    if (name == "wget"):
        path = path_dataset + 'wget/cache/' + 'graph{}'.format(str(id))
    elif (name == "streamspot"):
        path = path_dataset + 'streamspot/cache/' + 'graph{}'.format(str(id))
    elif (name == "SC2"):
        if (id < 125): path = path_dataset + 'SC2/cache/benign/' + 'graph{}'.format(str(id))
        else: path = path_dataset + 'SC2/cache/attack/' + 'graph{}'.format(str(id - 125))
    elif (name == 'Unicorn-Cadets'):
        if (id < 109): path = path_dataset + 'Unicorn-Cadets/cache/benign/' + 'graph{}'.format(str(id))
        else: path = path_dataset + 'Unicorn-Cadets/cache/attack/' + 'graph{}'.format(str(id - 109))
    elif (name == 'wget-long'):
        if (id < 100): path = path_dataset + 'wget-long/cache/benign/' + 'graph{}'.format(str(id))
        else: path = path_dataset + 'wget-long/cache/attack/' + 'graph{}'.format(str(id - 100))
    elif (name == 'clearscope-e3'):
        if (id < 44): path = path_dataset + 'clearscope-e3/cache/benign/' + 'graph{}'.format(str(id))
        else: path = path_dataset + 'clearscope-e3/cache/attack/' + 'graph{}'.format(str(id - 44))

    for fname in os.listdir(path):
        graphs.append(pkl.load(open(path + '/' + fname, 'rb')).to(device))

    return graphs