from tqdm import tqdm
from utils.dataloader import  load_entity_level_dataset
import torch




def entity_level_train(model,  snapshot, optimizer, max_epoch, device,  dataset_name, train_data):
    
        model = model.to(device)
        model.train()
        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in train_data:
                g = load_entity_level_dataset(dataset_name, 'train', i, snapshot, device)
                model.train()
                loss = model(g)
                loss /= len(train_data)
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))

        return model