import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=50):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        
        # Initialize weights
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()

    def forward(self, user, item):
        pred = (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        pred += self.user_biases(user) + self.item_biases(item)
        return pred.squeeze()

class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, layers=[64, 32, 16, 8]):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(n_items, layers[0] // 2)
        
        fc_layers = []
        for i in range(len(layers) - 1):
            fc_layers.append(nn.Linear(layers[i], layers[i+1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.1))
        
        self.fc_net = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.fc_net(x)
        return self.output_layer(x).squeeze()
