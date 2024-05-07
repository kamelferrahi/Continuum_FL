import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils.poolers import Pooling
from torch.nn import GRUCell
from .loss_func import sce_loss
from .gat import GAT
from .rnn import RNN_Cells
from utils.utils import create_norm
from functools import partial


class STGNN_AutoEncoder(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, device, number_snapshot, activation, feat_drop, negative_slope, residual, norm, pooling, loss_fn="sce", alpha_l=2, use_all_hidden = True):
        super(STGNN_AutoEncoder, self).__init__()

        
        #self.encoder = GNN_RNN(n_dim, e_dim, hidden_dim, out_dim ,n_layers, n_heads, device, number_snapshot)
        self.encoder = STGNN(n_dim, e_dim, out_dim, out_dim, n_layers, n_heads, n_heads, number_snapshot, activation, feat_drop, negative_slope, residual, norm, True, use_all_hidden, device)
        
        self.n_layers = n_layers
        self.pooler = Pooling(pooling)



        self.decoder = STGNN(out_dim, e_dim, out_dim, n_dim, 1, n_heads, 1, number_snapshot, activation, feat_drop, negative_slope, residual, norm, False, False, device)
        self.number_snapshot = number_snapshot

        if (use_all_hidden):
            self.encoder_to_decoder = nn.Linear(n_layers * out_dim, out_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(out_dim, out_dim, bias=False)

        self.use_all_hidden = use_all_hidden
        self.device = device
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        
        elif loss_fn == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "mae":
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError
        return criterion    
    

    def forward(self, g):   

        node_features = []
        new_t = []
        for G in g:
            new_g = G.clone()
            node_features.append(new_g.ndata['attr'].float())
            new_g.edata['attr'] = new_g.edata['attr'].float()
            new_t.append(new_g)
        final_embedding = self.encoder(new_t, node_features)
        encoding = []
        if (self.use_all_hidden):
            for i in range(len(g)):
                conca = [final_embedding[j][i] for j in range(len(final_embedding))]
                encoding.append(torch.cat(conca,dim=1))
        else:
            encoding = final_embedding[0]

        node_features = []
        for encoded in encoding:
            encoded = self.encoder_to_decoder(encoded)
            node_features.append(encoded)

        reconstructed = self.decoder(new_t, node_features)
        recon = reconstructed[0][-1]
        x_init = g[0].ndata['attr'].float()
        loss = self.criterion(recon, x_init)

        return loss
    
    def embed(self, g):
        node_features= []
        for G in g:
            node_features.append(G.ndata['attr'].float())
    
        return self.encoder.embed(g, node_features)
    

class STGNN(nn.Module):

    def __init__(self, input_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, n_heads_out, n_snapshot, activation, feat_drop, negative_slope, residual, norm, encoding, use_all_hidden, device):
        super(STGNN, self).__init__()

        if encoding:
            out = out_dim // n_heads
            hidden = out_dim // n_heads
        else:
            hidden = hidden_dim
            out = out_dim

        self.gnn = GAT(
            n_dim=input_dim,
            e_dim=e_dim,
            hidden_dim=hidden,
            out_dim=out,
            n_layers=n_layers,
            n_heads=n_heads,
            n_heads_out=n_heads_out,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=encoding,
        )
        self.rnn = RNN_Cells(out_dim, out_dim, n_snapshot, device)
        self.use_all_hidden = use_all_hidden

    def forward(self, G, node_features):

        embeddings = []
        for i in range(len(G)):
            g = G[i]
            if (self.use_all_hidden):
                node_embedding, all_hidden = self.gnn(g, node_features[i], return_hidden = self.use_all_hidden)
                embeddings.append(all_hidden)
                n_iter = len(all_hidden)

            else:
                embeddings.append(self.gnn(g, node_features[i], return_hidden = self.use_all_hidden))
                n_iter = 1
            
        result = []
        for j in range(n_iter):
            encoding = []

            for embedding in embeddings :
                if (self.use_all_hidden):
                    encoding.append(embedding[j])
                else:
                    encoding.append(embedding)

            result.append(self.rnn(encoding))

        return result
    

    def embed(self, G, node_features):
        embeddings = []
        for i in range(len(G)):
            g = G[i].clone()
            g.edata['attr'] = g.edata['attr'].float()
            embedding = self.gnn(g, node_features[i], return_hidden = False)
            embeddings.append(embedding)

        return self.rnn(embeddings)[-1]
