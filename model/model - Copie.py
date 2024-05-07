import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.poolers import Pooling
from dgl.nn import EdgeGATConv, GlobalAttentionPooling
from torch.nn import GRUCell
import dgl






class GNN_DTDG(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, device, mlp_layers, number_snapshot):
        super(GNN_DTDG, self).__init__()
        self.encoder = GNN_RNN(n_dim, e_dim, hidden_dim, out_dim ,n_layers, n_heads, device, number_snapshot)
        self.decoder = GNN_RNN(out_dim, e_dim, hidden_dim, n_dim ,n_layers, n_heads, device, number_snapshot)
        self.number_snapshot = number_snapshot
        self.classifier_layers = nn.ModuleList([
        ])

        for _ in range(mlp_layers - 1):
            self.classifier_layers.extend([
                nn.Linear(out_dim, out_dim).to(device),
                nn.ReLU(),
            ])
        self.classifier_layers.extend([
            nn.Linear(out_dim, 1).to(device),
            nn.Sigmoid()
        ])
        
        self.pooling_gate_nn = nn.Linear(out_dim , 1)
        self.pooling = GlobalAttentionPooling(self.pooling_gate_nn)
        self.pooler = Pooling("mean")
        self.encoder_to_decoder = nn.Linear( out_dim, out_dim, bias=False)
        
    def forward(self, g):   
        encoded = self.encoder(g)
        new_g = []
        i=  0
        for G in g:
            g_encoded = G.clone()
            g_encoded.ndata["attr"] = self.encoder_to_decoder(encoded[i])
            new_g.append(g_encoded)
            i+=1

        

        decoded = self.decoder(new_g)
        return decoded[-1]
       # x = self.pooler(G, embeddings, [1])[0] 
       # h_g = x.clone()
      #  for layer in self.classifier_layers:
       #     x = layer(x)
    
    def embed(self, g):
        return self.encoder(g)[-1]



class GNN_RNN(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, device,  number_snapshot):
        super(GNN_RNN, self).__init__()
        self.device = device
        self.gnn_layers = nn.ModuleList([EdgeGATConv(in_feats=n_dim, edge_feats=e_dim, out_feats=out_dim, num_heads=n_heads, allow_zero_in_degree=True).to(device)])
        
        self.out_dim = out_dim
        
        for _ in range(n_layers-1):
            self.gnn_layers.append(
            EdgeGATConv(in_feats=out_dim, edge_feats=e_dim, out_feats=out_dim, num_heads=n_heads, allow_zero_in_degree=True).to(device)
            )
        
        self.rnn_layers = nn.ModuleList([])

        for _ in range(number_snapshot):
            self.rnn_layers.append(
                GRUCell(out_dim, out_dim, device = device)
            )
        
        self.classifier_layers = nn.ModuleList([
        ])


    def forward(self, g):        
        i = 0
        H_s = []
        for G in g:
            
            with G.to(self.device).local_scope():
                x = G.ndata["attr"].float()
                e = G.edata["attr"].float()
                for layer in self.gnn_layers:
                    r = layer(G, x, e)
                    x = torch.mean(r,dim=1).to(self.device)
                    del r

                #if ( i == 0):
               #     H = self.rnn_layers[i](x, x)
               # else:
              #      H = self.rnn_layers[i](x, H)
                
                H = x
                H_s.append(H)
                embeddings = H.clone()
                i+=1
                #x = self.pooling(g[0], x)[0]

        
        return H_s