import numpy as np
from tqdm import tqdm
import torch
from utils.dataloader import load_graph
import matplotlib.pyplot as plt
from model.eval import batch_level_evaluation

def batch_level_train(model,  train_loader, optimizer, max_epoch, device, n_dim, e_dim, dataset_name, validation=True):
    
    epoch_iter = tqdm(range(max_epoch))
    model.to(device)  # Move model to GPU
    n_epoch = 0
    validation_f1 = []
    loss_global = []
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for iter, batch in enumerate(train_loader):
            batch_g = [load_graph(int(idx), dataset_name, device) for idx in batch]  # Move data to GPU
            model.train()
            g = batch_g[0]
            loss = model(g) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g, g

        n_epoch +=1

        if (validation):
            validation_f1.append(batch_level_evaluation(model, model.pooler, device, ['knn'], dataset_name, n_dim, e_dim)[0])

        loss_global.append(np.mean(loss_list))
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    if (validation):
        plt.plot(list(range(n_epoch)), validation_f1, label='Graph 2', marker='o', linestyle='-')
        plt.plot(list(range(n_epoch)), loss_global, label='Graph 2', marker='x', linestyle='--')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Two Graphs on the Same Plot')

        # Add a legend
        plt.legend()

        # Display the plot
        plt.show()
    return model
