import numpy as np
from tqdm import tqdm
from utils.dataloader import load_metadata, load_entity_level_dataset
from model.test import test
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model.eval import batch_level_evaluation



def entity_level_train(model,  snapshot, optimizer, max_epoch, device,  dataset_name, train_data):
    
        model = model.to(device)
        model.train()
        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in train_data:
                print(i)
                g = load_entity_level_dataset(dataset_name, 'train', i, snapshot, device)
                model.train()
                loss = model(g)

               # loss_dest = torch.ones( len(g[0].ndata["attr"]), n_dim, device=device)

                #for h in range(len(g)):
               # G = g[0]
               # for i in range(len(G.ndata["attr"])):
               #     for j in range(len(G.ndata["attr"][i])):
             #           loss_dest[i][j] = G.ndata["attr"][i][j]


               # loss = loss_func(pred , loss_dest)
                print(loss)
                loss /= len(train_data)
                print(loss)
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))

        return model