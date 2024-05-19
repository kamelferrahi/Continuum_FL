import logging

import fedml
from utils.dataloader import load_partition_data, load_data, load_metadata, darpa_split
from fedml import FedMLRunner
from trainer.magic_trainer import MagicTrainer
from trainer.magic_aggregator import MagicWgetAggregator
from model.model import STGNN_AutoEncoder
from utils.config import build_args
from trainer.magic_trainer import MagicTrainer
from trainer.magic_aggregator import MagicWgetAggregator



def generate_dataset(name, number, nsnapshot):
    (
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
    ) = load_partition_data(number, name, nsnapshot) 
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        len(train_data_global),
    ]
    
    if (name in ['wget', 'streamspot', 'SC2', 'Unicorn-Cadets', 'wget-long', 'clearscope-e3']):
        
        return dataset, load_data(name)
    else:
        return dataset, load_metadata(name) 
           
    
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    # init device

    device = fedml.device.get_device(args)
    dataset_name = args.dataset
    number = args.client_num_in_total
    nsnapshot = args.snapshot
    dataset, metadata = generate_dataset(dataset_name, number, nsnapshot)
    main_args = build_args()
    if (dataset_name in ['wget', 'streamspot', 'SC2', 'Unicorn-Cadets', 'wget-long', 'clearscope-e3']):
            
        main_args.max_epoch = 6
        out_dim = 64
        if (dataset_name == 'SC2'):
            gnn_layer = 3
        else:
            gnn_layer = 5
        n_node_feat = metadata['n_feat']
        n_edge_feat = metadata['e_feat']
        use_all_hidden =  True
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
    else: 
        use_all_hidden =  False
        n_node_feat = metadata['node_feature_dim']
        n_edge_feat = metadata['edge_feature_dim']
            #train_index = [104, 118, 86, 74, 16, 12, 117, 108, 59, 146, 97, 49, 107, 47, 23, 111, 32, 124, 121, 119, 141, 50, 43, 98, 73, 80, 4, 140, 1, 17, 55, 136, 95, 120, 103, 94, 34, 68, 130, 26, 30, 29, 129, 71, 6, 128, 84, 85, 72, 96, 87, 58, 81, 79, 31, 37, 54, 93, 135, 33, 61, 134, 52, 106, 126, 139, 8, 115, 82, 46, 101, 114, 60, 138, 132, 5, 2, 19, 143, 77, 92, 123, 42, 113, 125, 15, 105, 14, 145, 148]
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        main_args.max_epoch = 50
        out_dim = 128

        if (dataset_name == 'cadets-e3'):
            gnn_layer = 4
        else:
            gnn_layer = 5

        

    model = STGNN_AutoEncoder(main_args.n_dim, main_args.e_dim, out_dim, out_dim, gnn_layer, 4, device,  nsnapshot, 'prelu', 0.1, main_args.negative_slope, True, 'BatchNorm', main_args.pooling, alpha_l=main_args.alpha_l, use_all_hidden=use_all_hidden).to(device)  # Move model to GPU
    #train_single(main_args, model, data)
    trainer = MagicTrainer(model, args, dataset_name)
    aggregator = MagicWgetAggregator(model, args, dataset_name)
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
    # start training
    #darpa_split("theia")
