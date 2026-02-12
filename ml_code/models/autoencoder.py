import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        
        self.embedding_dim = kwargs['embed_dim']
        self.seq_dim = kwargs['seq_len']
        vocab_size = kwargs['vocab_size']
        # embedding dimension is required
            
        self.embedding_layer = nn.Embedding(vocab_size, self.embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=32, kernel_size=5, stride=1, padding=0)  # output shape: (32, 196)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) # output shape: (32, 98)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0) # output shape: (64, 94)
        # another pooling layer --> output shape: (64, 47)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 47, 512)
        self.fc2 = nn.Linear(512, self.seq_dim * self.embedding_dim)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.permute(0, 2, 1)         
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), self.seq_dim, self.embedding_dim)

        return x