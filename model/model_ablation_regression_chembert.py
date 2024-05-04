import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool,GATConv,SAGEConv,Sequential,GraphNorm,global_max_pool,DeepGCNLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_output_layers=1):
        super(Net,self).__init__()
        self.n_output_layers = n_output_layers
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        self.outputlayer = nn.Sequential(
            nn.Linear(767, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.n_output_layers),
        )
    def forward(self, inputs):
        outputs = self.model(**inputs)
        x = outputs[0]
        x = x[:, -1, :]
        x = self.outputlayer(x)
        return x


if __name__ == '__main__':
    x=torch.randn(3,9)
    edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
    drug = ["CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)C(NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)O"
            "C(=O)[C@H](NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC"
            "1=O)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C"]
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    inputs = tokenizer(drug, padding=True, truncation=True, return_tensors="pt")


    model = Net()

    data1 = Data(x=x, edge_index=edge_index, id=1)
    data_list = [data1]
    loader = DataLoader(data_list,batch_size=1)
    for i, data in enumerate(loader):
        print(data)
        outputs = model.forward(data, inputs)
        print(outputs)

