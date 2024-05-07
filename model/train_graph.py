import numpy as np
from tqdm import tqdm
from utils.dataloader import load_graph
from model.test import test
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model.eval import batch_level_evaluation

def batch_level_train(model, label, train_loader, validation_index, optimizer, max_epoch, device, n_dim, e_dim, dataset_name):
    
    epoch_iter = tqdm(range(max_epoch))
    loss_func = nn.CrossEntropyLoss()
    model.to(device)  # Move model to GPU
    n_epoch = 0
    validation_f1 = []
    loss_global = []
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for iter, batch in enumerate(train_loader):
            print(iter)
            labels = [[label[idx]] for idx in batch]
            batch_g = [load_graph(int(idx), dataset_name, device) for idx in batch]  # Move data to GPU
            model.train()
            g = batch_g[0]
            loss = model(g) 
            
            
            #loss_dest = torch.ones(len(g.ndata["attr"]), n_dim, device=device)

           #   for i in range(len(g.ndata["attr"])):
             #     for j in range(len(g.ndata["attr"][i])):
              #        loss_dest[i][j] = g.ndata["attr"][i][j]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g, g

        n_epoch +=1
        #validation_f1.append(test(model, label, validation_index, device, n_dim, e_dim, dataset_name)[0])
       # validation_f1.append(batch_level_evaluation(model, model.pooler, device, ['knn'], dataset_name, n_dim, e_dim)[0])

        loss_global.append(np.mean(loss_list))
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    
    return model
