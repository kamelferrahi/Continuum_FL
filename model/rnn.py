from torch.nn import GRUCell
import torch.nn as nn 


class RNN_Cells(nn.Module):   
    def __init__(self, input_dim, hidden_dim, n_cells, device) :
        super(RNN_Cells, self).__init__()
        self.cells = nn.ModuleList()
        
        for i in range(n_cells):
            self.cells.append(GRUCell(input_dim, hidden_dim, device=device))


    def forward(self, inputs):

        results = []
        for i in range(len(self.cells)):
            if (i == 0):
                results.append(self.cells[i](inputs[i], inputs[i]))
            else:
                results.append(self.cells[i](inputs[i], results[i-1]))

        return results