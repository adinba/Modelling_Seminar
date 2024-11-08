import torch
from torch_geometric.data import Data, InMemoryDataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        
        if data_list is not None:
            self.data, self.slices = self.collate(self.data_list)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.data_list is not None:
            self.data, self.slices = self.collate(self.data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])

    def len(self):
        return self.data.num_graphs if self.data else 0

    def get(self, idx):
        data = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(slice(slices[idx], slices[idx + 1]))
            data[key] = item[s]
        return data

    def save(self):
        torch.save((self.data, self.slices), self.processed_paths[0])

    @staticmethod
    def load(root):
        return CustomGraphDataset(root=root)

# Usage
loaded_dataset = CustomGraphDataset.load('./data_set_toy')

for i in loaded_dataset:
    print(i)