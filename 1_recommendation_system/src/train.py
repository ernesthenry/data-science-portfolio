import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from model import MatrixFactorization

class MovieLensDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def train_model():
    # Mock data for demonstration (real implementation would load MovieLens)
    print("Generating mock data...")
    n_users, n_items = 1000, 500
    n_samples = 5000
    
    users = np.random.randint(0, n_users, n_samples)
    items = np.random.randint(0, n_items, n_samples)
    ratings = np.random.uniform(1, 5, n_samples)
    
    dataset = MovieLensDataset(users, items, ratings)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MatrixFactorization(n_users, n_items)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in loader:
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "models/recommender_mf.pth")

if __name__ == "__main__":
    train_model()
