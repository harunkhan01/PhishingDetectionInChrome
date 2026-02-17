import torch

class TrancoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx]