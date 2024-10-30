import torch
import torch.nn as nn
import torch_geometric.nn as NN
import pickle
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import sys


modified = True
root_file = "./graph_datasets/qm9"
root_file2 = "./graph_datasets/QM9.pt"
if modified == False:
    data = QM9(root_file)
else:
    data = torch.load(root_file2)
torch.manual_seed(300)

train_split = 1e5 / len(data)
validation_split = 18e3 / len(data)
test_split = 1.0 - train_split - validation_split

if modified == False:
    y_data = torch.empty((1, 12))
    for i in range(len(data)):
        y_data = torch.cat([y_data, data[i].y[:, :12]])

    mean = y_data[1:].mean(dim = 0)
    absolute_deviation = torch.mean(torch.abs(y_data[1:] - mean), dim = 0)
    for i in range(len(data)):
        data[i].y[0, :12] = (data[i].y[0, :12] - mean) / absolute_deviation
        
    torch.save(data, root_file2)


data = data.shuffle()

# original_data = QM9(root_file)
# org_train_data, org_val_data, org_test_data = random_split(original_data, [train_split, validation_split, test_split])
train_data, val_data, test_data = random_split(data, [train_split, validation_split, test_split])

data_train = DataLoader(train_data, batch_size = 96, shuffle = True)

class EGNN_Layer(MessagePassing):
    def __init__(self, gconv_in_dim, hidden_dim):
        super().__init__(aggr='add')
        self.EdgeMLP = nn.Sequential(
            nn.Linear(2 * gconv_in_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.NodeMLP = nn.Sequential(
            nn.Linear(gconv_in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, h, edge_index):

        new_h = self.propagate(edge_index, x = x, h = h)

        return new_h

    def message(self, x_i, x_j, h_i, h_j):
        distance = torch.sum((x_i - x_j)**2, dim = 1, keepdim=True)

        edge_features = self.EdgeMLP(torch.cat([h_i, h_j, distance], dim = 1))
        
        return edge_features

    def update(self, aggr_out, h):
        new_h = self.NodeMLP(torch.cat([h, aggr_out], dim = 1))

        return new_h

class EGNN_Model(nn.Module):
    def __init__(self, lr, weight_decay, checkpoint_file, gconv_in_dim, hidden_dim, nr_layers, loss_fn, nr_classes):
        super().__init__()

        self.final_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, nr_classes),
        )

        self.checkpoint_file = checkpoint_file
        self.loss_fn = loss_fn
        self.layer = nn.ModuleList()
        self.nr_layers = nr_layers

        self.layer.append(EGNN_Layer(gconv_in_dim=gconv_in_dim, hidden_dim=hidden_dim))

        for i in range(nr_layers - 1):
            self.layer.append(EGNN_Layer(gconv_in_dim=hidden_dim, hidden_dim=hidden_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay=weight_decay)

    def save_checkpoint(self):
        torch.save({
            "model_state_dict":self.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict()
        }, self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        self.load_state_dict(chkpt["model_state_dict"])
        self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])

    def forward(self, x, h, edge_index, batch=None):
        for i in range(self.nr_layers):
            h = self.layer[i](x, h, edge_index)
        
        if batch != None:
            h = self.final_mlp2(NN.pool.global_add_pool(self.final_mlp1(h), batch=batch))
        else:
            h = self.final_mlp2(NN.pool.global_add_pool(self.final_mlp1(h), batch=None))

        return h
    
model = EGNN_Model(lr=1e-3, 
                   weight_decay=1e-16, 
                   checkpoint_file="./EGNN_checkpts/Checkpoint.pth",
                   gconv_in_dim=next(iter(data_train)).x.shape[1], 
                   hidden_dim=128, 
                   nr_layers=5,
                   loss_fn=nn.MSELoss(), 
                   nr_classes=12)

model.load_checkpoint()

def validation_test():
    total_loss = 0
    with torch.no_grad():
        for i in range(len(val_data)):
            y = model(val_data[i].pos, val_data[i].x, val_data[i].edge_index).squeeze()
            total_loss += model.loss_fn(y, val_data[i].y[0, :12])
        
        return total_loss / len(val_data)

def test():
    with torch.no_grad():
        r_y = torch.empty((1, 12))
        for i in range(len(test_data)):
            graph = test_data[i]
            y = model(graph.pos, graph.x, graph.edge_index)
            r_y = torch.cat([r_y, y])

        mean = torch.mean(r_y[1:], dim = 0)
        std = torch.mean(torch.abs(r_y[1:] - mean), dim = 0)

        std[2:5] = std[2:5] * 1e3
        std[6:11] = std[6:11] * 1e3

        print(std)

def train(epochs):
    for i in tqdm(range(epochs)):
        for _, data in tqdm(enumerate(data_train)):
            y = model(data.pos, data.x, data.edge_index, batch=data.batch)

            loss = model.loss_fn(y, data.y[:, :12])

            loss.backward()

            model.optimizer.step()
            model.optimizer.zero_grad()

        if i % 5 == 0:
            print(f"The total loss at epoch {i} is:{validation_test()}")
            model.save_checkpoint()
